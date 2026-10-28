"""Ray ownership and materialization of RL image/video training payloads."""

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import ray
import torch

from xtuner.v1.data_proto.rl_data import MultimodalInfo, RolloutState
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.rl.utils.ray_utils import free_object_refs


_PAYLOAD_FIELDS = ("pixel_values", "pixel_values_videos")


def put_multimodal_info(info: MultimodalInfo | None) -> None:
    """Store media once before the sampler repeats a prompt.

    Args:
        info (MultimodalInfo | None): Training media; updated in place with shared Ray references.
    """
    if info is not None:
        payloads = cast(dict[str, Any], info)
        for name in _PAYLOAD_FIELDS:
            value = payloads.get(name)
            if value is not None and not isinstance(value, ray.ObjectRef):
                payloads[name] = ray.put(value)


def resolve_multimodal_inputs(seq_ctx: SequenceContext) -> None:
    """Materialize packed media on CPU before device transfer and SP splitting.

    Args:
        seq_ctx (SequenceContext): Context containing tensors, arrays, or packed Ray references.
    """
    for name in _PAYLOAD_FIELDS:
        value = getattr(seq_ctx, name)
        if value is None or isinstance(value, torch.Tensor):
            continue
        if isinstance(value, (ray.ObjectRef, np.ndarray)):
            value = [value]
        # Batch the object-store fetch; a pack can contain many media refs, including shared repeats.
        refs = [item for item in value if isinstance(item, ray.ObjectRef)]
        fetched = iter(ray.get(refs) if refs else [])
        tensors = [torch.as_tensor(next(fetched) if isinstance(item, ray.ObjectRef) else item) for item in value]
        setattr(seq_ctx, name, tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0))


def free_multimodal_inputs(contexts: Iterable[SequenceContext]) -> None:
    """Release media references after all training workers have finished the
    batch.

    A batch owns media for one training consumption. Released rollout states must not be replayed again.

    Args:
        contexts (Iterable[SequenceContext]): Packed batches sharing sampler-owned media references.
    """
    refs = []
    for seq_ctx in contexts:
        for name in _PAYLOAD_FIELDS:
            value = getattr(seq_ctx, name)
            if isinstance(value, ray.ObjectRef):
                refs.append(value)
            elif isinstance(value, list):
                refs.extend(item for item in value if isinstance(item, ray.ObjectRef))
    free_object_refs(refs)


def validate_image_only_rollout(state: RolloutState) -> None:
    """Reject video inputs unsupported by the built-in online rollout
    transports.

    Args:
        state (RolloutState): Request whose visual training inputs must match generation inputs.
    """
    has_video = state.extra_fields.get("video_data") or (
        state.mm_info is not None and state.mm_info.get("pixel_values_videos") is not None
    )
    if has_video:
        raise NotImplementedError(
            "Online video rollout/teacher scoring is not supported by the built-in transports. "
            "Use image/text prompts online; video multimodal inputs are supported for training replay only."
        )
