"""Shared image plans and lazy resolution; native tokenization stays in each family."""

import copy
import hashlib
import importlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import torch


def _enabled(name):
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def lazy_media_enabled():
    return _enabled("XTUNER_LAZY_MEDIA")


def lazy_media_shared_decode():
    return _enabled("XTUNER_LAZY_MEDIA_SHARED")


def make_image_plan(processor, paths, grids, *, oss_loader_cfg=None):
    """A processor adapter needs from_dict/to_dict and images -> pixels/grid calls.

    The importable processor class is the extension point; no family switch or
    live processor/image is carried through dataloader IPC. Geometry is predicted
    by the native tokenizer and checked against the real processor on consumption.
    """
    if not paths or len(paths) != len(grids):
        raise ValueError("lazy images require one predicted grid per source")
    config = processor.to_dict()
    json.dumps(config)  # Reject live objects in the descriptor.
    cls = type(processor)
    if "<locals>" in cls.__qualname__ or not hasattr(cls, "from_dict"):
        raise ValueError("lazy processor must be importable and support from_dict")
    return {
        "processor_class": f"{cls.__module__}:{cls.__qualname__}",
        "processor_config": config,
        "images": [{"path": str(path), "grid": list(grid)} for path, grid in zip(paths, grids)],
        "patch_dim": 3 * processor.temporal_patch_size * processor.patch_size**2,
        "oss_loader_cfg": oss_loader_cfg,
        "media_dtype": torch.float32,
    }


def merge_image_plans(plans):
    if not plans:
        return None
    first = {key: value for key, value in plans[0].items() if key not in ("images", "_future")}
    for plan in plans:
        if "_future" in plan:
            raise ValueError("merge plans before scheduling media prefetch")
        if {key: value for key, value in plan.items() if key != "images"} != first:
            raise ValueError("media plans in one pack must share processor, dtype and source settings")
    return {**first, "images": [entry for plan in plans for entry in plan["images"]]}


def pack_image_plans(items, *, media_dtype=None):
    """Merge retained records, resolving mixed eager/lazy records in native order."""
    for item in items:
        if item.get("media_plan") is not None and media_dtype is not None:
            item["media_plan"] = dict(item["media_plan"], media_dtype=media_dtype)
    if any(item.get("media_plan") is not None for item in items) and any(
        item.get("pixel_values") is not None for item in items
    ):
        for item in items:
            plan = item.pop("media_plan", None)
            if plan is not None:
                item["pixel_values"], item["image_grid_thw"] = materialize_image_plan(plan)
    return merge_image_plans([item["media_plan"] for item in items if item.get("media_plan") is not None])


def materialize_image_plan(plan):
    """Process images independently in order, closing decoded sources on failure too."""
    module_name, class_name = plan["processor_class"].split(":", 1)
    processor_class = importlib.import_module(module_name)
    for name in class_name.split("."):
        processor_class = getattr(processor_class, name)
    local = threading.local()

    def process(entry):
        from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import load_image
        from xtuner.v1.datasets.mllm_tokenize_fn.qwen3_vl_utils import pil_loader
        from xtuner.v1.datasets.utils import apply_exif_orientation

        if not hasattr(local, "processor"):
            local.processor = processor_class.from_dict(copy.deepcopy(plan["processor_config"]))
        path = entry["path"]
        if "s3://" in path:
            config = plan.get("oss_loader_cfg")
            if not config or config["backend"] != "petrel":
                raise ValueError("lazy OSS image requires petrel oss_loader_cfg")
            if not hasattr(local, "client"):
                from petrel_client.client import Client

                local.client = Client(**config["backend_kwargs"])
            image = pil_loader(local.client.get(path))
        else:
            image = load_image(path)
        oriented = None
        try:
            oriented = apply_exif_orientation(image)
            result = local.processor(images=[oriented], return_tensors="pt")
            pixels, grid = result["pixel_values"], result["image_grid_thw"]
            expected = torch.tensor([entry["grid"]], dtype=torch.long)
            if pixels.device.type != "cpu" or grid.device.type != "cpu" or not torch.equal(grid, expected):
                raise ValueError("lazy media processor must return CPU tensors matching the predicted grid")
            rows = int(expected.prod())
            if tuple(pixels.shape) != (rows, plan["patch_dim"]):
                raise ValueError("lazy media pixel shape differs from predicted patches")
            return pixels.to(plan["media_dtype"]), grid
        finally:
            if oriented is not None and oriented is not image:
                oriented.close()
            image.close()

    threads = int(os.environ.get("XTUNER_LAZY_MEDIA_THREADS", "1"))
    if not 0 <= threads <= 64:
        raise ValueError("XTUNER_LAZY_MEDIA_THREADS must be between 0 and 64")
    if threads <= 1:
        parts = [process(entry) for entry in plan["images"]]
    else:
        with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="lazy-image") as pool:
            parts = list(pool.map(process, plan["images"]))
    return torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts])


def resolve_lazy_media(context):
    """Called explicitly by a family's visual-input preparation, never a base hook."""
    import torch.distributed as dist

    from xtuner.v1.train.lazy_media_prefetch import shared_materialize, take_media_future

    plan = context.media_plan
    mesh = context.sequence_parallel_mesh
    group = mesh.get_group() if mesh is not None and mesh.size() > 1 and lazy_media_shared_decode() else None
    if group is not None:
        # Includes EMPTY records: divergent per-rank plans fail before payload calls.
        stable_plan = None if plan is None else {key: value for key, value in plan.items() if key != "_future"}
        digest = hashlib.sha256(json.dumps(stable_plan, sort_keys=True, default=str).encode()).digest()
        signature = torch.tensor(list(digest), dtype=torch.uint8, device=context.input_ids.device)
        signatures = [torch.empty_like(signature) for _ in range(dist.get_world_size(group))]
        dist.all_gather(signatures, signature, group=group)
        if any(not torch.equal(signature, other) for other in signatures):
            raise ValueError("SP ranks have different lazy media plans")
    if plan is None:
        return context
    if context.pixel_values is not None:
        raise ValueError("lazy plan and eager pixels are mutually exclusive")
    if group is not None:
        pixels, grid = shared_materialize(plan, group, context.input_ids.device, plan["patch_dim"])
        pixels = pixels.cpu()  # Preserve native CPU pixel placement, including Qwen SP slicing.
    else:
        future = take_media_future(plan)
        pixels, grid = future.result() if future is not None else materialize_image_plan(plan)
    return context.copy(media_plan=None, pixel_values=pixels, image_grid_thw=grid.to(context.input_ids.device))
