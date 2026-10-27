"""CPU integration checks for the leader-only loader and the shared transport path.

Uses the real Trainer methods and XT JSONL/packing/sampler/checkpoint code.
Gloo here validates CPU behavior; these tests do not validate CUDA/NCCL execution.
"""

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.datasets import DataloaderConfig, DatasetConfig, FTDPTokenizeFnConfig
from xtuner.v1.train import sp_batch_broadcast as transport
from xtuner.v1.train.toy_tokenizer import UTF8ByteTokenizer
from xtuner.v1.train.trainer import Trainer


def _config(data_dir, workers, enabled=True):
    return DataloaderConfig(
        dataset_config_list=[
            {
                "dataset": DatasetConfig(anno_path=str(data_dir)),
                "tokenize_fn": FTDPTokenizeFnConfig(max_length=128),
            }
        ],
        pack_max_length=128,
        pack_chunk_size=10,
        pack_workers=1,
        num_workers=workers,
        sp_media_dedup=enabled,
    )


def _trainer(mesh, cfg, steps=None, epoch=1, cur_step=0):
    obj = Trainer.__new__(Trainer)
    obj.data_mesh, obj.sp_mesh = mesh, mesh["sp"]
    obj.tokenizer = UTF8ByteTokenizer()
    obj._global_batch_size, obj._micro_batch_size = 4, None
    obj._total_step, obj._total_epoch = steps, 2 if steps is None else None
    obj._cur_step, obj._cur_epoch = cur_step, epoch
    # Execute the actual data-initialization section of __init__, without constructing
    # a model/optimizer. Do not add a production wrapper just to accommodate this test.
    import ast
    import inspect
    import textwrap

    from xtuner.v1.train import trainer as trainer_module

    nodes = ast.parse(textwrap.dedent(inspect.getsource(Trainer.__init__))).body[0].body
    start = next(
        i
        for i, node in enumerate(nodes)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_sp_media_dedup" for t in node.targets)
    )
    end = next(i for i in range(start + 1, len(nodes)) if isinstance(nodes[i], ast.Assert))
    section = ast.Module(body=nodes[start:end], type_ignores=[])
    namespace = dict(vars(trainer_module), self=obj, dataloader_cfg=cfg, seed=17, total_step=steps)
    exec(compile(section, "<Trainer.__init__:data initialization>", "exec"), namespace)
    return obj


def _snapshot(batch):
    return [
        (
            item["seq_ctx"].input_ids.tolist(),
            item["shifted_labels"].tolist(),
            item["seq_ctx"].position_ids.tolist(),
            item["seq_ctx"].cu_seq_lens_q.tolist(),
        )
        for item in batch
    ]


def _take(trainer, iterator, count, rng_states=None):
    out = []
    for _ in range(count):
        out.append(_snapshot(next(iterator)))
        if rng_states is not None:
            rng_states.append(torch.get_rng_state())
        trainer._cur_step += 1
    return out


def _save(trainer, path):
    trainer._save_dataloader(path)
    # Original Trainer train_state stores these values separately from data state.
    state = (trainer._cur_step, trainer._cur_epoch)
    dist.barrier()
    return state


def _loader_worker(rank, store, data_dir, workers):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=4, timeout=timedelta(seconds=120)
    )
    mesh = init_device_mesh("cpu", (2, 2, 1), mesh_dim_names=("dp", "sp", "tp"))
    cfg = _config(data_dir, workers)
    torch.manual_seed(314)
    trainer = _trainer(mesh, cfg)
    assert (trainer._dataloader is None) == (rank % 2 == 1)
    length = trainer.total_step // 2
    assert trainer.total_step == length * 2 and length >= 3
    # Keep a full continuous reference beyond the second checkpoint's resume point.
    trainer._total_step = length + 6
    stream = trainer._data_iter()
    continuous_rng = []
    prefix = _take(trainer, stream, length - 1, continuous_rng)
    first_meta = _save(trainer, Path(data_dir) / "first.pt")
    suffix = _take(trainer, stream, 7, continuous_rng)

    # Fresh instance -> actual mid-epoch load -> consume through the epoch boundary.
    resumed = _trainer(mesh, cfg, length + 6, epoch=first_meta[1], cur_step=first_meta[0])
    resumed._resume_dataloader(Path(data_dir) / "first.pt")
    resumed_stream = resumed._data_iter()
    assert _take(resumed, resumed_stream, 3) == suffix[:3]
    second_meta = _save(resumed, Path(data_dir) / "second.pt")
    assert resumed._cur_epoch == trainer._cur_epoch == 2
    if rank == 0:
        state = torch.load(Path(data_dir) / "second.pt", map_location="cpu")
        assert state["total_consumed_samples"] == (length + 2) * 4

    # Save from the resumed branch, reload again, and actually consume four more batches.
    again = _trainer(mesh, cfg, length + 6, epoch=second_meta[1], cur_step=second_meta[0])
    again._resume_dataloader(Path(data_dir) / "second.pt")
    assert _take(again, again._data_iter(), 4) == suffix[3:]

    # BASE/off-path: same original pipeline on every rank, full ordered values and RNG.
    torch.manual_seed(314)
    baseline = _trainer(mesh, _config(data_dir, workers, enabled=False), length + 6)
    assert baseline._dataloader is not None
    baseline_stream = baseline._data_iter()
    baseline_rng = []
    assert _take(baseline, baseline_stream, length + 6, baseline_rng) == prefix + suffix
    assert all(torch.equal(a, b) for a, b in zip(continuous_rng, baseline_rng))
    baseline._save_dataloader(Path(data_dir) / "baseline.pt")
    if rank == 0:
        first = torch.load(Path(data_dir) / "first.pt", map_location="cpu")
        assert first["total_consumed_samples"] == (length - 1) * 4

    # Genuinely test the SP1 fallback through the production initialization method.
    one_mesh = init_device_mesh("cpu", (4, 1, 1), mesh_dim_names=("dp", "sp", "tp"))
    single = _trainer(one_mesh, cfg, steps=1)
    assert single._dataloader is not None and not single._sp_media_dedup
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("workers", [0, 4])
def test_real_loader_resume(tmp_path, workers):
    from itertools import chain

    source = Path(__file__).parents[1] / "resource/openai_sft.jsonl"
    samples = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in chain.from_iterable([samples] * 5)))
    mp.spawn(_loader_worker, args=(str(tmp_path / "store"), str(data_dir), workers), nprocs=4, join=True)


def _media_batch(tmp_dir):
    from transformers import AutoTokenizer
    from xtuner.v1.datasets.collator import glm53_vl_sft_collator
    from xtuner.v1.datasets.mllm_tokenize_fn import Glm53VLTokenizeFnConfig

    model = os.environ["XTUNER_TEST_GLM53_MODEL"]
    fn = Glm53VLTokenizeFnConfig(processor_path=model).build(
        AutoTokenizer.from_pretrained(model, trust_remote_code=True),
        anno_name="leader-loader-review",
    )
    fn.state = "runtime"
    image = Path(tmp_dir) / "image.png"
    clip = Path(tmp_dir) / "video"
    content = [
        {"type": "image", "image": {"url": str(image), "image_wh": [224, 224]}},
        {
            "type": "video",
            "video": {"url": str(clip), "image_wh": [224, 224], "origin_video_length": 6, "origin_fps": 2.0},
        },
        {"type": "text", "text": "Describe these."},
    ]
    # BASE supports an image sample and a video sample in the same pack; do not
    # invent an unsupported mixed-media source sample for this transport test.
    image_item = fn(
        {
            "messages": [
                {"role": "user", "content": [content[0], content[2]]},
                {"role": "assistant", "content": "An image."},
            ]
        }
    )
    video_item = fn(
        {
            "messages": [
                {"role": "user", "content": [content[1], content[2]]},
                {"role": "assistant", "content": "A clip."},
            ]
        }
    )
    text = fn({"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]})
    return glm53_vl_sft_collator(
        [[image_item, video_item], [text], [video_item]],
        pack_max_length=8192,
        padding_token_idx=0,
        image_token_id=fn.processor.image_token_id,
        merge_unit=fn.merge_unit,
    )


def _transport_worker(rank, store, media_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    reference = _media_batch(media_dir)
    # New fields, arbitrary shapes/dtypes, nested tensors and aliases need no registry edit.
    reference[0]["seq_ctx"].extra_features = {
        "nested": [torch.arange(24, dtype=torch.float64).reshape(2, 3, 4), torch.empty((0, 7), dtype=torch.int16)],
        "ordinary": (None, True, "copied metadata", 42),
    }
    reference[0]["extra_batch_field"] = {"mask": torch.tensor([True, False])}
    got, epoch, draw = transport.broadcast_batch(
        reference if rank == 0 else None, dist.group.WORLD, epoch=3, seed_draw=True
    )
    assert epoch == 3 and draw

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert type(a) is type(b) and len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        elif hasattr(a, "cu_seq_lens_q"):
            for key, value in vars(a).items():
                if key not in ("device", "sequence_parallel_mesh"):
                    equal(value, getattr(b, key))
        else:
            assert a == b

    equal(reference, got)
    assert got[0]["seq_ctx"].cu_seq_lens_q is got[0]["seq_ctx"].cu_seq_lens_k
    assert got[0]["seq_ctx"].cu_seq_lens_q_cpu is got[0]["seq_ctx"].cu_seq_lens_q
    dist.destroy_process_group()


@pytest.mark.skipif(
    not os.environ.get("XTUNER_TEST_GLM53_MODEL"),
    reason="Set XTUNER_TEST_GLM53_MODEL to a local GLM53 tokenizer/processor directory",
)
def test_media_and_dynamic_transport(tmp_path):
    Image.new("RGB", (224, 224), (10, 90, 150)).save(tmp_path / "image.png")
    (tmp_path / "video").mkdir()
    for i in range(6):
        Image.new("RGB", (224, 224), (i * 20, 90, 150)).save(tmp_path / "video" / f"{i:05d}.jpg")
    mp.spawn(_transport_worker, args=(str(tmp_path / "store"), str(tmp_path)), nprocs=2, join=True)


def _failure_worker(rank, store):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2, timeout=timedelta(seconds=10)
    )
    # An unsupported leaf raises at its source. The process launcher terminates peers;
    # no per-error all_reduce or recovery path is expected anymore.
    transport.broadcast_batch({"unsupported": object()} if rank == 0 else None, dist.group.WORLD)


def test_direct_failure_terminates_workers(tmp_path):
    from torch.multiprocessing.spawn import ProcessRaisedException

    with pytest.raises(ProcessRaisedException):
        mp.spawn(_failure_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True)
