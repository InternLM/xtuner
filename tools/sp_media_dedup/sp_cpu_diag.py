#!/usr/bin/env python3
"""CC8 SP media-dedup CPU diagnosis harness.

Runs the NATIVE xtuner data chain (DataloaderConfig.build -> ExpandSoftPackDataset ->
LengthGroupedSampler(dp_mesh) -> fork DataLoader workers -> Glm53VLTokenizeFunction ->
glm53_vl_sft_collator) under a CPU/Gloo device mesh shaped exactly like the trainer's
`_init_data_mesh` (dp, sp, tp), with dp_mesh == mesh["dp"] passed to the loader build —
the same object the trainer passes (trainer.py:703-707).

Deviations from the real training path (recorded, not hidden):
  - device mesh device is "cpu" and backend gloo (trainer uses cuda/nccl when not offloading);
    the data path only consumes mesh["dp"], whose size/local_rank semantics are identical.
  - no model weights, no CUDA, no optimizer, no training step. The model-side split
    (SequenceContext.split / vision patch sharding) is static-audit evidence only.
  - optional diagnostic-only shrink of pack_max_length / window to bound memory; any
    deviation from the native params is written into every event header.

Instrumentation (monkeypatch, applied before dataset build so fork workers inherit):
  oss_read    : Qwen3VLOSSLoader client.get  (path, blake2b(content), nbytes)
  decode      : pil_loader / load_image      (source bytes/path sha, PIL size/mode)
  exif        : apply_exif_orientation       (count)
  processor   : image_processor.__call__     (per-image input sizes; output pixel_values
                                              blake2b, shape, dtype, nbytes, grid_thw)
  sample_mm   : multi_modal_get_item         (data_id, input_ids sha, per-sample pixel sha)
  deliver     : main loop per yielded pack   (seq_ctx input_ids/pixel/grid sha, nbytes)
  phase       : lifecycle markers for the external PSS/USS monitor to join against

Events: JSON lines appended to EVENTS_DIR/r{RANK}-pid{PID}.jsonl (one file per process,
no interleaving). blake2b used for large blobs (fast, cryptographic enough for identity).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # hard CPU mode; no CUDA init anywhere

EVENTS_DIR = None
_RANK = None
_WRITE_LOCK = threading.Lock()
_TLS = threading.local()


def _media_sink():
    """Thread-local list collecting (kind, path, sha) during one sample's processing."""
    return getattr(_TLS, "media", None)


def blake_bytes(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=16).hexdigest()


def blake_tensor(t) -> tuple[str, int, str]:
    import torch

    t = t.contiguous()
    n = t.numel() * t.element_size()
    h = hashlib.blake2b(digest_size=16)
    view = t.view(torch.uint8) if t.dtype != torch.uint8 else t
    h.update(view.numpy().tobytes())
    return h.hexdigest(), n, str(t.dtype)


def emit(kind: str, **fields) -> None:
    rec = {"ts": time.time(), "rank": _RANK, "pid": os.getpid(), "ppid": os.getppid(), "kind": kind}
    rec.update(fields)
    line = json.dumps(rec, ensure_ascii=False)
    path = os.path.join(EVENTS_DIR, f"r{_RANK}-pid{os.getpid()}.jsonl")
    with _WRITE_LOCK:
        with open(path, "a") as f:
            f.write(line + "\n")
            f.flush()


def install_instrumentation(model_path: str) -> dict:
    """Monkeypatch the four media touchpoints. Returns a status dict for the header."""
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    import xtuner.v1.datasets.mllm_tokenize_fn.glm53_vl_tokenize_fn as g53
    from xtuner.v1.datasets.mllm_tokenize_fn.glm53_vl_tokenize_fn import Glm53VLTokenizeFunction
    from xtuner.v1.datasets.mllm_tokenize_fn.qwen3_vl_utils import Qwen3VLOSSLoader

    status = {}

    # --- OSS read (petrel client.get on the Qwen3VLOSSLoader singleton) --------------------
    _orig_loader_init = Qwen3VLOSSLoader.__init__

    def _patched_loader_init(self, *a, **kw):
        _orig_loader_init(self, *a, **kw)
        if getattr(self, "_cc8_patched", False):
            return
        self._cc8_patched = True
        orig_get = self.client.get

        def recording_get(path, *a2, **kw2):
            data = orig_get(path, *a2, **kw2)
            sha = blake_bytes(bytes(data))
            emit("oss_read", path=str(path), sha=sha, nbytes=len(data))
            sink = _media_sink()
            if sink is not None:
                sink.append(("oss_read", str(path), sha))
            return data

        try:
            self.client.get = recording_get
            status["oss_read_patch"] = "ok"
        except Exception as e:  # pragma: no cover
            status["oss_read_patch"] = f"failed: {e!r}"

    Qwen3VLOSSLoader.__init__ = _patched_loader_init

    # --- decode: pil_loader (OSS bytes) and load_image (local path) ------------------------
    _orig_pil = g53.pil_loader

    def recording_pil(data):
        img = _orig_pil(data)
        sha = blake_bytes(bytes(data))
        emit("decode", src="bytes", sha=sha, nbytes=len(data),
             width=img.size[0], height=img.size[1], mode=img.mode)
        sink = _media_sink()
        if sink is not None:
            sink.append(("decode", None, sha))
        return img

    g53.pil_loader = recording_pil
    status["pil_loader_patch"] = "ok"

    _orig_load = g53.load_image

    def recording_load(path):
        img = _orig_load(path)
        try:
            with open(path, "rb") as f:
                raw = f.read()
            sha, nbytes = blake_bytes(raw), len(raw)
        except OSError:
            sha, nbytes = None, None
        emit("decode", src="path", path=str(path), sha=sha, nbytes=nbytes,
             width=img.size[0], height=img.size[1], mode=img.mode)
        sink = _media_sink()
        if sink is not None and sha:
            sink.append(("decode", str(path), sha))
        return img

    g53.load_image = recording_load
    status["load_image_patch"] = "ok"

    _orig_exif = g53.apply_exif_orientation

    def recording_exif(image):
        out = _orig_exif(image)
        emit("exif", width=out.size[0], height=out.size[1])
        return out

    g53.apply_exif_orientation = recording_exif
    status["exif_patch"] = "ok"

    # --- processor (class-level wrap; tokenize_fn deepcopies share the class) --------------
    probe = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    proc_cls = type(probe.image_processor)
    status["processor_class"] = f"{proc_cls.__module__}.{proc_cls.__name__}"
    _orig_proc_call = proc_cls.__call__

    def recording_proc_call(self, images=None, **kw):
        out = _orig_proc_call(self, images=images, **kw)
        in_sizes = [(im.size[0], im.size[1]) for im in (images or [])]
        pv = out.get("pixel_values") if isinstance(out, dict) else getattr(out, "pixel_values", None)
        grid = out.get("image_grid_thw") if isinstance(out, dict) else getattr(out, "image_grid_thw", None)
        rec = {"n_images": len(in_sizes), "in_sizes": in_sizes}
        if pv is not None:
            sha, nbytes, dtype = blake_tensor(pv)
            rec.update({"pixel_sha": sha, "pixel_nbytes": nbytes, "pixel_dtype": dtype,
                        "pixel_shape": list(pv.shape)})
        if grid is not None:
            rec["grid_thw"] = grid.tolist()
        emit("processor", **rec)
        return out

    proc_cls.__call__ = recording_proc_call
    status["processor_patch"] = "ok"

    # --- per-sample runtime items (Glm53VLDataItem is a TypedDict -> dict) ------------------
    _orig_mm = Glm53VLTokenizeFunction.multi_modal_get_item

    def recording_mm(self, data_item, media_root=""):
        _TLS.media = []
        out = _orig_mm(self, data_item, media_root)
        media = _TLS.media
        _TLS.media = None
        rec = {"data_id": data_item.get("id"),
               "input_ids_sha": blake_bytes(json.dumps(out["input_ids"]).encode()),
               "n_tokens": out["num_tokens"], "n_imgs": out["num_imgs"],
               "media_shas": sorted({m[2] for m in media if len(m) > 2 and m[2]})}
        prev = getattr(_TLS, "pack_media", None)
        if prev is not None:  # accumulate pack-level media list (nw=0 path)
            prev.extend(rec["media_shas"])
        if out.get("pixel_values") is not None:
            sha, nbytes, dtype = blake_tensor(out["pixel_values"])
            rec.update({"pixel_sha": sha, "pixel_nbytes": nbytes, "pixel_dtype": dtype})
        emit("sample_mm", **rec)
        return out

    Glm53VLTokenizeFunction.multi_modal_get_item = recording_mm
    status["sample_mm_patch"] = "ok"

    _orig_vt = Glm53VLTokenizeFunction.video_get_item

    def recording_vt(self, data_item, media_root=""):
        out = _orig_vt(self, data_item, media_root)
        emit("sample_video", data_id=data_item.get("id"), n_videos=out.get("num_videos"))
        return out

    Glm53VLTokenizeFunction.video_get_item = recording_vt
    status["sample_video_patch"] = "ok"

    _orig_pt = Glm53VLTokenizeFunction.pure_text_get_item

    def recording_pt(self, data_item):
        out = _orig_pt(self, data_item)
        emit("sample_text", data_id=data_item.get("id"),
             input_ids_sha=blake_bytes(json.dumps(out["input_ids"]).encode()),
             n_tokens=out.get("num_tokens"))
        return out

    Glm53VLTokenizeFunction.pure_text_get_item = recording_pt
    status["sample_text_patch"] = "ok"

    _ = torch  # keep import local
    return status


def _self_pss_uss_kb():
    out = {}
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                parts = line.split()
                if parts and parts[0] in ("Pss:", "Private_Clean:", "Private_Dirty:"):
                    out[parts[0].rstrip(":")] = int(parts[1])
    except OSError:
        pass
    return out


TENSOR_FIELDS = ("input_ids", "cu_seq_lens_q", "cu_seq_lens_k", "position_ids", "mm_token_type_ids",
                 "pixel_values", "image_grid_thw")


def serialize_seq_ctx(seq):
    """Pack the trainer-consumed seq_ctx state into (spec, blob) for D2 sharing."""
    import numpy as np

    fields, blobs, off = [], [], 0
    scalars = {"num_padding": getattr(seq, "num_padding", 0)}
    for name in TENSOR_FIELDS:
        t = getattr(seq, name, None)
        if t is None:
            continue
        t = t.contiguous()
        b = t.numpy().tobytes()
        fields.append({"name": name, "dtype": str(t.dtype), "shape": list(t.shape),
                       "offset": off, "nbytes": len(b)})
        blobs.append(b)
        off += len(b)
    nit = getattr(seq, "num_img_tokens", None)
    if nit is not None:
        scalars["num_img_tokens"] = nit
    return {"fields": fields, "scalars": scalars}, b"".join(blobs)


def rebuild_seq_ctx(spec, blob, seq_cls):
    """Reconstruct seq_ctx tensors from the shared blob (zero-copy for pixel_values)."""
    import numpy as np

    tensors = {}
    for f in spec["fields"]:
        arr = np.frombuffer(blob, dtype=f["dtype"], count=f["nbytes"] // np.dtype(f["dtype"]).itemsize,
                            offset=f["offset"])
        t = torch.from_numpy(arr.reshape(f["shape"]))
        tensors[f["name"]] = t
    field_names = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "block_table",
                   "position_ids", "seq_idx", "image_grid_thw", "deepstack_visual_embeds",
                   "visual_pos_masks", "pixel_values", "inputs_embeds", "num_img_tokens",
                   "pixel_values_videos", "video_grid_thw", "mm_token_type_ids",
                   "rollout_routed_experts")
    kwargs = {n: tensors.get(n) for n in field_names if n in tensors}
    if "num_img_tokens" not in kwargs and "num_img_tokens" in spec["scalars"]:
        kwargs["num_img_tokens"] = spec["scalars"]["num_img_tokens"]
    seq = seq_cls(
        input_ids=tensors.get("input_ids"),
        cu_seq_lens_q=tensors["cu_seq_lens_q"], cu_seq_lens_k=tensors["cu_seq_lens_k"],
        max_length_q=int(tensors["cu_seq_lens_q"][0, -1].item()) if "max_length_q" not in tensors else tensors.get("max_length_q"),
        max_length_k=int(tensors["cu_seq_lens_k"][0, -1].item()) if "max_length_k" not in tensors else tensors.get("max_length_k"),
        num_padding=spec["scalars"].get("num_padding", 0),
        sequence_parallel_mesh=None, block_table=tensors.get("block_table"), device="cpu",
        **kwargs,
    )
    return seq


class D2Session:
    """Per-pack D2-minimal flow (plan §一.D): producer publishes the pack's collated
    pixels once; consumers attach read-only, verify bitwise identity vs their own
    native collate, then HOLD the shared view in place of their local copy (native
    tensor dropped), and release at the pack boundary. Barriers per pack prove order
    and group boundaries. Consumer CPU work is NOT skipped in D2-min (that short-
    circuit inside multi_modal_get_item is D2-full; recorded honestly). Modes:
    verify | producer_fail (publish skipped -> consumer bounded-timeout fallback)
    | tiny_budget (publish refuses over budget -> all-local)."""

    def __init__(self, cfg, emit, dist):
        import sys as _sys

        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from sp_share import PackShare, SegmentMeta

        self._SegmentMeta = SegmentMeta
        self.emit, self.dist = emit, dist
        self.rank = _RANK
        self.mode = cfg.get("d2_mode", "off")
        self.run_id = cfg.get("d2_run_id", "run")
        budget = int(cfg.get("d2_budget_bytes", 4 << 30))
        if self.mode == "tiny_budget":
            budget = 1
        self.share = PackShare(f"cc8sp-{self.run_id}", budget)
        self.timeout_s = float(cfg.get("d2_timeout_s", 15))
        self.lease_s = float(cfg.get("d2_lease_s", 300))
        self.held = None  # consumer's current shared view
        emit("phase", name="d2_start", mode=self.mode, budget_bytes=budget,
             usage_before=self.share.current_usage())

    def publish(self, rec):
        if self.rank != 0 or self.mode == "producer_fail":
            return
        if not rec.get("pixel_nbytes"):
            return
        meta = self._SegmentMeta(
            run_id=self.run_id, pack_idx=rec["pack_idx"], producer_pid=os.getpid(),
            content_sha16=rec["pixel_sha"], media_sha_list=rec.get("media_shas", "[]"),
            cfg_sha16=rec.get("cfg_sha16", ""), shape=rec.get("pixel_shape", ()),
            dtype=rec.get("pixel_dtype", ""), payload_nbytes=rec["pixel_nbytes"],
            grid_shape=rec.get("grid_shape", ()), grid_nbytes=rec.get("grid_nbytes", 0))
        ok = self.share.try_publish(f"p{rec['pack_idx']}", meta, rec["pixel_bytes"],
                                    rec.get("grid_bytes", b""), self.lease_s)
        self.emit("d2_publish", pack_idx=rec["pack_idx"], ok=ok,
                  payload_nbytes=rec["pixel_nbytes"], usage=self.share.current_usage())

    def consume(self, rec):
        """Attach (bounded wait), verify, hold. Returns True if view held."""
        if self.rank == 0 or not rec.get("pixel_nbytes"):
            return False
        t0 = time.time()
        view = self.share.attach(f"p{rec['pack_idx']}", timeout_s=self.timeout_s,
                                 expect_content_sha=rec["pixel_sha"])
        attach_s = round(time.time() - t0, 3)
        if view is None:
            self.emit("d2_attach", pack_idx=rec["pack_idx"], ok=False, attach_s=attach_s,
                      fallback="native_local_processing")
            return False
        t = view.pixel_tensor()
        sha = hashlib.blake2b(t.contiguous().view(torch.uint8).numpy().tobytes(),
                              digest_size=16).hexdigest()
        match = sha == rec["pixel_sha"]
        self.emit("d2_attach", pack_idx=rec["pack_idx"], ok=True, attach_s=attach_s,
                  readback_sha_match=match, meta_content_sha=view.meta.content_sha16,
                  meta_shape=list(view.meta.shape), payload_nbytes=rec["pixel_nbytes"])
        self.held = view
        return match

    def swap_measure(self, rec):
        """Drop-native/hold-view memory measurement: caller already del'd the native
        batch; sample self memory while the shared view is the only copy."""
        if self.held is None:
            return
        self.emit("d2_hold", pack_idx=rec["pack_idx"], mem_pss_uss_kb=_self_pss_uss_kb(),
                  payload_nbytes=rec["pixel_nbytes"])

    def release_pack(self, rec):
        if self.held is not None:
            mem_before_release = _self_pss_uss_kb()
            self.share.release(self.held)
            self.held = None
            self.emit("d2_release", pack_idx=rec["pack_idx"],
                      mem_before_release_kb=mem_before_release,
                      mem_after_release_kb=_self_pss_uss_kb())

    def finish(self):
        if self.rank == 0:
            deadline = time.time() + 10
            while self.share.current_usage() > 0 and time.time() < deadline:
                time.sleep(0.2)
            swept = self.share.sweep_own()
            self.emit("phase", name="d2_end", usage_after=self.share.current_usage(), swept=swept)
        else:
            self.emit("phase", name="d2_end")


def main() -> int:
    global EVENTS_DIR, _RANK
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="diagnosis run config JSON")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    EVENTS_DIR = cfg["events_dir"]
    os.makedirs(EVENTS_DIR, exist_ok=True)

    import torch
    import torch.distributed as dist

    torch.set_num_threads(int(cfg.get("torch_threads", 4)))
    _RANK = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    deadline = time.time() + float(cfg.get("deadline_s", 420))

    def watchdog():
        delay = deadline - time.time()
        if delay > 0:
            time.sleep(delay)
        emit("timeout", deadline_s=cfg.get("deadline_s", 420))
        os._exit(97)

    threading.Thread(target=watchdog, daemon=True).start()

    # --- config fingerprint + identity -------------------------------------------------------
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    proc_cfg_path = os.path.join(cfg["model_path"], "processor_config.json")
    proc_sha = hashlib.sha256(open(proc_cfg_path, "rb").read()).hexdigest()[:16] \
        if os.path.isfile(proc_cfg_path) else "missing"
    emit("identity", cfg_sha=cfg_sha, processor_cfg_sha=proc_sha,
         xtuner_sha=cfg.get("xtuner_sha", ""), cmdline=sys.argv, role="rank-main")

    # --- gloo pg + trainer-shaped mesh -------------------------------------------------------
    t0 = time.time()
    dist.init_process_group(backend="gloo")
    tp = int(cfg.get("tp_size", 1))
    sp = int(cfg["sp_size"])
    assert world % (sp * tp) == 0, f"world {world} not divisible by sp*sp {sp}*{tp}"
    dp = world // (sp * tp)
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

    mesh: DeviceMesh = init_device_mesh("cpu", (dp, sp, tp), mesh_dim_names=("dp", "sp", "tp"))
    emit("phase", name="mesh_ready", dp=dp, sp=sp, tp=tp, elapsed=round(time.time() - t0, 3))

    # --- instrumentation BEFORE dataset build -------------------------------------------------
    inst_status = install_instrumentation(cfg["model_path"])
    emit("phase", name="instrumented", status=inst_status)

    # --- native build (mirrors trainer.py:703) -----------------------------------------------
    from transformers import AutoTokenizer

    from xtuner.v1.datasets import Glm53VLTokenizeFnConfig
    from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
    from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    emit("phase", name="tokenizer_ready")

    ds_cfg = DatasetConfig(
        name=cfg.get("dataset_name", "cc8_diag"),
        anno_path=cfg["anno_path"],
        class_name="VLMJsonlDataset",
        media_root=cfg["media_root"],
        sample_ratio=1.0,
        enable_sequential_sampler=True,
        cache_dir=cfg["cache_dir"],
        cache_tag=cfg["cache_tag"],
    )
    tk_cfg = Glm53VLTokenizeFnConfig(
        processor_path=cfg["model_path"],
        max_length=int(cfg["sample_max_length"]),
        max_pixels=int(cfg["max_pixels"]),
        min_pixels=None,
        llm_pack_weight=-3.2,
        visual_pack_weight=5.0,
        max_frames=24,
        oss_loader_cfg=OSSLoaderConfig(backend_kwargs={"conf_path": cfg["ceph_conf"]})
        if cfg.get("ceph_conf") else None,
    )
    dl_cfg = DataloaderConfig(
        dataset_config_list=[{"dataset": ds_cfg, "tokenize_fn": tk_cfg}],
        collator="glm53_vl_sft_collator",
        collator_kwargs={"image_token_id": int(cfg["image_token_id"]),
                         "merge_unit": int(cfg["merge_unit"])},
        pack_level="soft",
        pack_max_length=int(cfg["pack_max_length"]),
        pack_chunk_size=int(cfg["pack_chunk_size"]),
        pack_workers=int(cfg["pack_workers"]),
        global_pack=True,
        group_by_length=True,
        num_workers=int(cfg["num_workers"]),
    )
    emit("phase", name="dataset_build_start")
    loader = dl_cfg.build(
        tokenizer=tokenizer, dp_mesh=mesh["dp"], global_batch_size=1, micro_batch_size=1,
        seed=int(cfg["seed"]),
    )
    emit("phase", name="dataset_build_end", elapsed=round(time.time() - t0, 3))

    # --- consume packs_to_consume packs -------------------------------------------------------
    n_packs = int(cfg["packs_to_consume"])
    emit("phase", name="iterate_start", packs_to_consume=n_packs)
    _TLS.pack_media = []
    it = iter(loader)
    delivered = 0
    d2 = D2Session(cfg, emit, dist) if cfg.get("d2_mode", "off") != "off" else None
    for k in range(n_packs):
        t_pack = time.time()
        try:
            batch = next(it)
        except StopIteration:
            emit("phase", name="iterator_exhausted", packs_available=k)
            break
        items = batch if isinstance(batch, list) else [batch]
        for j, data in enumerate(items):
            seq = data["seq_ctx"]
            rec = {"pack_idx": k, "item_idx": j, "n_samples": len(items),
                   "cfg_sha16": cfg_sha}
            if seq.input_ids is not None:
                sha, nbytes, dtype = blake_tensor(seq.input_ids)
                rec.update({"input_ids_sha": sha, "input_ids_nbytes": nbytes,
                            "input_ids_len": int(seq.input_ids.shape[-1])})
            if seq.pixel_values is not None:
                sha, nbytes, dtype = blake_tensor(seq.pixel_values)
                rec.update({"pixel_sha": sha, "pixel_nbytes": nbytes, "pixel_dtype": dtype,
                            "pixel_shape": list(seq.pixel_values.shape)})
                if d2 is not None and _RANK == 0:  # only the producer needs the bytes view
                    rec["pixel_bytes"] = seq.pixel_values.contiguous().numpy().data
                    if seq.image_grid_thw is not None:
                        g = seq.image_grid_thw.contiguous()
                        rec["grid_bytes"] = g.numpy().tobytes()
                        rec["grid_shape"] = list(g.shape)
                        rec["grid_nbytes"] = g.numel() * g.element_size()
            if seq.image_grid_thw is not None:
                rec["grid_sha"] = blake_bytes(seq.image_grid_thw.contiguous().view(torch.uint8).numpy().tobytes())
                rec["n_images_in_pack"] = int(seq.image_grid_thw.shape[0])
            rec["media_shas"] = json.dumps(sorted(getattr(_TLS, "pack_media", []) or []))
            rec["deliver_s"] = round(time.time() - t_pack, 3)
            emit("deliver", **{k2: v for k2, v in rec.items() if k2 != "pixel_bytes"})
        delivered += 1
        _TLS.pack_media = []
        # --- D2-minimal per-pack flow (order & group boundary via barriers) --------------
        if d2 is not None and rec.get("pixel_nbytes"):
            d2.publish(rec)          # rank0 only; no-op on consumers
            dist.barrier()           # publication visible to all
            if d2.rank != 0:
                d2.consume(rec)      # bounded attach + bitwise verify + hold view
            dist.barrier()           # all ranks attached or fell back
            del batch, items
            gc.collect()             # native copies dropped (consumers keep view only)
            if d2.rank != 0:
                d2.swap_measure(rec)
                emit("phase", name=f"pack_{k}_held_shared_view")
            else:
                emit("phase", name=f"pack_{k}_delivered")
            dist.barrier()           # measurement done everywhere
            d2.release_pack(rec)     # consumers release view; refcount may unlink
        else:
            emit("phase", name=f"pack_{k}_delivered")
            del batch, items
            gc.collect()
        emit("phase", name=f"pack_{k}_released")
    if d2 is not None:
        try:
            d2.finish()
        except Exception as e:  # D2 bookkeeping must never take the round down
            emit("d2_error", err=repr(e))

    # --- teardown: workers must exit before we report success ---------------------------------
    del it, loader
    gc.collect()
    emit("phase", name="loader_released", packs_delivered=delivered)
    dist.barrier()
    emit("phase", name="final_barrier_done")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
