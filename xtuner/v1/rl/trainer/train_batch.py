"""Worker-local training batch: the single complete CPU sample source.

``TrainBatch`` owns the one-time next-token shift, the CPU sample records and the
pack materialization round-trip. The training worker only holds a
``TrainBatch | None``; roles and consumers borrow it through method arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Literal, Sequence, cast

import numpy as np
import ray
import torch

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model.compose.base import BaseComposeConfig

from .pack import RankPlan, TokenBatchProto


if TYPE_CHECKING:
    from xtuner.v1.model.base import TransformerConfig
    from xtuner.v1.rl.distillation import DistillationConfig, DistillationTrainerAdapter

    from .worker import TrainBatchAttr


_UINT16_CAPACITY = 1 << 16


@dataclass
class TrainSample:
    """CPU training tensors of one rollout sample.

    ``values``, ``returns`` and ``advantages`` start as ``None`` and are filled by
    later stages. The source ``seq_ctx`` keeps object refs unresolved; each pack
    resolves them again from the pristine cache.
    """

    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor
    rollout_logprobs: torch.Tensor | None
    teacher_targets: dict[str, torch.Tensor]
    reward_score: float | None = None
    values: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    advantages: torch.Tensor | None = None


@dataclass(frozen=True)
class FullPack:
    """One packed CPU training unit.

    ``indices`` address ``TrainBatch`` samples, ``lengths`` are their real
    (unpadded) token counts and ``full_data`` holds the complete packed
    ``seq_ctx`` plus the aligned loss fields.
    """

    indices: tuple[int, ...]
    lengths: tuple[int, ...]
    full_data: dict


class TrainBatch:
    """Validate, materialize and round-trip one local training batch.

    Args:
        states (list[RolloutState]): This DP rank's rollout states, in plan order.
        rank_plan (RankPlan): Local pack indices addressing ``states``.
        attrs (TrainBatchAttr): Batch-level attributes shared by all workers.
        teacher_adapter (DistillationTrainerAdapter): Rollout teacher target adapter.
        pack_max_length (int): Token capacity of one pack.
        model_cfg (TransformerConfig | BaseComposeConfig): Model config used for
            routed-expert padding.
        teacher_loss_keys (list[str]): Extra loss fields required on every pack.
        distillation_config (DistillationConfig | None): Distillation config used to
            size teacher padding fields.
    """

    def __init__(
        self,
        states: list[RolloutState],
        rank_plan: RankPlan,
        attrs: TrainBatchAttr,
        teacher_adapter: DistillationTrainerAdapter,
        *,
        pack_max_length: int,
        model_cfg: TransformerConfig | BaseComposeConfig,
        teacher_loss_keys: list[str],
        distillation_config: DistillationConfig | None,
    ) -> None:
        self.plan = rank_plan
        self.attrs = attrs
        self.pack_max_length = pack_max_length
        self.model_cfg = model_cfg
        self.teacher_loss_keys = teacher_loss_keys
        self.distillation_config = distillation_config
        self._samples: list[TrainSample] = [
            self._build_sample(state, attrs["use_3d_position_ids"], teacher_adapter) for state in states
        ]

    def materialize(self, indices: tuple[int, ...]) -> FullPack:
        """Concatenate one pack's CPU samples and tail-pad it to capacity.

        Args:
            indices (tuple[int, ...]): Local sample indices of the pack.

        Returns:
            FullPack: The packed CPU sequence context and loss fields.
        """
        samples = [self._samples[index] for index in indices]
        lengths = tuple(sample.shifted_labels.shape[1] for sample in samples)
        padding_len = self.pack_max_length - sum(lengths)
        if padding_len < 0:
            raise ValueError(f"Pack tokens {sum(lengths)} exceed pack_max_length {self.pack_max_length}")

        fields: dict[str, torch.Tensor | None] = {}
        if samples:
            fields["shifted_labels"] = torch.cat([sample.shifted_labels for sample in samples], dim=1)
            advantages = [sample.advantages for sample in samples]
            if any(value is None for value in advantages) and not all(value is None for value in advantages):
                raise ValueError("advantages must be all-present or all-absent within a pack")
            if all(value is not None for value in advantages):
                fields["advantages"] = torch.cat([cast(torch.Tensor, value) for value in advantages], dim=1)
            else:
                fields["advantages"] = None
            rollout_logprobs = [sample.rollout_logprobs for sample in samples]
            if "rollout_logprobs" in self._loss_keys():
                if not all(value is not None for value in rollout_logprobs):
                    raise ValueError("Required pack field rollout_logprobs must be present on every sample")
                fields["rollout_logprobs"] = torch.cat(
                    [cast(torch.Tensor, value) for value in rollout_logprobs], dim=1
                )
            else:
                fields["rollout_logprobs"] = None
            for name, attribute in (("returns", "returns"), ("old_values", "values")):
                rows = [getattr(sample, attribute) for sample in samples]
                if any(row is not None for row in rows) and not all(row is not None for row in rows):
                    raise ValueError(f"{name} must be all-present or all-absent within a pack")
                fields[name] = (
                    torch.cat([cast(torch.Tensor, row) for row in rows], dim=1)
                    if all(row is not None for row in rows)
                    else None
                )
            for key in self.teacher_loss_keys:
                values = [sample.teacher_targets.get(key) for sample in samples]
                if not all(value is not None for value in values):
                    raise ValueError(f"Required pack field {key} must be present on every sample: {key}")
                fields[key] = torch.cat([cast(torch.Tensor, value) for value in values], dim=1)
        else:
            # Scheduling-only padding slot: the whole pack materializes as padding.
            fields["shifted_labels"] = self._padding_loss_input("shifted_labels", padding_len)
            fields["advantages"] = self._padding_loss_input("advantages", padding_len)
            fields["rollout_logprobs"] = (
                self._padding_loss_input("rollout_logprobs", padding_len)
                if "rollout_logprobs" in self._loss_keys()
                else None
            )
            fields["returns"] = self._padding_loss_input("returns", padding_len) if self.has_targets else None
            fields["old_values"] = self._padding_loss_input("old_values", padding_len) if self.has_targets else None
            for key in self.teacher_loss_keys:
                fields[key] = self._padding_loss_input(key, padding_len)

        if samples and padding_len > 0:
            fields = {
                key: (
                    torch.cat([cast(torch.Tensor, value), self._padding_loss_input(key, padding_len)], dim=1)
                    if value is not None
                    else None
                )
                for key, value in fields.items()
            }

        seq_ctx_list = [sample.seq_ctx for sample in samples]
        if padding_len > 0:
            seq_ctx_list.append(self._padding_seq_ctx(padding_len))
        seq_ctx = SequenceContext.cat(seq_ctx_list)
        assert seq_ctx.input_ids is not None
        seq_len = seq_ctx.input_ids.shape[1]
        for key, value in fields.items():
            if value is None:
                continue
            assert value.shape[1] == seq_len, f"{key}: {value.shape[1]} vs {seq_len}"
        return FullPack(indices, lengths, {"seq_ctx": seq_ctx, **fields})

    def unpack_rows(self, pack: FullPack, full_rows: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
        """Split full pack rows back into per-sample rows.

        Callers must already have removed any sequence-parallel padding; only pack
        padding is trimmed here.

        Args:
            pack (FullPack): The pack the rows belong to.
            full_rows (torch.Tensor): Rows covering the full pack.

        Returns:
            list[tuple[int, torch.Tensor]]: ``(sample_index, row)`` pairs.
        """
        rows = full_rows[:, : sum(pack.lengths)].detach().cpu()
        output: list[tuple[int, torch.Tensor]] = []
        offset = 0
        for sample_index, length in zip(pack.indices, pack.lengths):
            output.append((sample_index, rows[:, offset : offset + length]))
            offset += length
        return output

    def install_advantages(self, payload: list[float] | TokenBatchProto) -> None:
        """Broadcast scalar advantages or install aligned token rows.

        Args:
            payload (list[float] | TokenBatchProto): One scalar per sample, or one
                shifted-token row per sample.

        Returns:
            None
        """
        rows = self._expand_rows(payload)
        for sample, row in zip(self._samples, rows):
            sample.advantages = row.masked_fill(sample.shifted_labels.eq(-100), 0.0)

    def install_old_values(self, rows: list[torch.Tensor]) -> None:
        """Install the complete frozen critic values of every sample.

        Args:
            rows (list[torch.Tensor]): One full token row per sample, in plan order.

        Returns:
            None
        """
        self._validate_full_rows(rows, "old_values")
        for sample, row in zip(self._samples, rows):
            sample.values = row.detach().reshape(1, -1).float().cpu()

    def install_ppo_targets(self, advantages: TokenBatchProto, returns: TokenBatchProto) -> None:
        """Install paired full-token PPO targets atomically.

        Both payloads are validated before any sample is written. Advantages are
        masked at ignored positions; returns keep their raw values.

        Args:
            advantages (TokenBatchProto): One full token row per sample.
            returns (TokenBatchProto): One full token row per sample.

        Returns:
            None
        """
        adv_rows = advantages.chunk()
        ret_rows = returns.chunk()
        self._validate_full_rows(adv_rows, "advantages")
        self._validate_full_rows(ret_rows, "returns")
        for sample, adv, ret in zip(self._samples, adv_rows, ret_rows):
            labels = sample.shifted_labels
            sample.advantages = adv.reshape_as(labels).masked_fill(labels.eq(-100), 0.0)
            sample.returns = ret.reshape_as(labels)

    def trajectory_data(self) -> dict[str, torch.Tensor]:
        """Return complete CPU trajectory data for the numerical estimator.

        No pack padding and no Ray handles are involved; empty local batches
        still return a valid ``cu_seq_lens`` so every rank can join the global
        reduction.

        Returns:
            dict[str, torch.Tensor]: ``values``, ``scores``, ``mask`` and
            ``cu_seq_lens`` over local samples.
        """
        if not self._samples:
            return {
                "values": torch.zeros(0, dtype=torch.float32),
                "scores": torch.zeros(0, dtype=torch.float32),
                "mask": torch.zeros(0, dtype=torch.bool),
                "cu_seq_lens": torch.zeros(1, dtype=torch.int32),
            }
        values: list[torch.Tensor] = []
        scores: list[float] = []
        masks: list[torch.Tensor] = []
        lengths: list[int] = []
        for sample in self._samples:
            if sample.values is None:
                raise ValueError("trajectory_data requires complete old_values")
            if sample.reward_score is None:
                raise ValueError("Segment GAE requires a reward score for every sample")
            flat_values = sample.values.detach().float().reshape(-1).cpu()
            flat_labels = sample.shifted_labels.reshape(-1).cpu()
            if flat_values.numel() != flat_labels.numel():
                raise ValueError("old_values length does not match shifted labels")
            values.append(flat_values)
            scores.append(float(sample.reward_score))
            masks.append(flat_labels.ne(-100))
            lengths.append(flat_labels.numel())
        cu_seq_lens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
        cu_seq_lens[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
        return {
            "values": torch.cat(values),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "mask": torch.cat(masks),
            "cu_seq_lens": cu_seq_lens,
        }

    def require_targets(self, role: Literal["actor", "critic"]) -> None:
        """Fail fast when a role's targets are incomplete.

        Args:
            role (Literal["actor", "critic"]): Role whose required fields are checked.

        Returns:
            None
        """
        fields = ("advantages",) if role == "actor" else ("old_values", "returns")
        for sample in self._samples:
            for name in fields:
                # values are stored as ``values`` on the record; targets keep their public names.
                attribute = "values" if name == "old_values" else name
                if getattr(sample, attribute) is None:
                    raise ValueError(f"{role} requires complete {name} rows")

    def _validate_full_rows(self, rows: list[torch.Tensor], field: str) -> None:
        if len(rows) != len(self._samples):
            raise ValueError(f"{field} rows {len(rows)} do not match samples {len(self._samples)}")
        for index, (sample, row) in enumerate(zip(self._samples, rows)):
            if row.numel() != sample.shifted_labels.numel():
                raise ValueError(
                    f"{field} row {index} has {row.numel()} tokens, expected {sample.shifted_labels.numel()}"
                )

    def close(self) -> None:
        """Release the local CPU samples; remote refs are owned by the
        trainer."""
        self._samples.clear()

    @property
    def has_targets(self) -> bool:
        """Whether the batch already carries PPO targets (returns/old
        values)."""
        return bool(self._samples) and all(
            sample.returns is not None and sample.values is not None for sample in self._samples
        )

    def local_masked_token_count(self, packs: list[list[int]]) -> int:
        """Count trainable tokens of the given local packs before
        materialization.

        Args:
            packs (list[list[int]]): Local pack indices.

        Returns:
            int: Local masked-token count.
        """
        return sum(int(self._samples[index].shifted_labels.ne(-100).sum()) for pack in packs for index in pack)

    def _loss_keys(self) -> list[str]:
        keys = list(self.attrs["pack_loss_keys"])
        for key in self.teacher_loss_keys:
            if key not in keys:
                keys.append(key)
        return keys

    def _expand_rows(self, payload: list[float] | TokenBatchProto) -> list[torch.Tensor]:
        if isinstance(payload, TokenBatchProto):
            rows = payload.chunk()
            if len(rows) != len(self._samples):
                raise ValueError(f"rows {len(rows)} do not match samples {len(self._samples)}")
            expanded: list[torch.Tensor] = []
            for sample, row in zip(self._samples, rows):
                flat = row.detach().reshape(-1).float()
                if flat.numel() != sample.shifted_labels.shape[1]:
                    raise ValueError(
                        f"Token advantage length mismatch: {flat.numel()} does not match "
                        f"shifted labels length {sample.shifted_labels.shape[1]}"
                    )
                expanded.append(flat.unsqueeze(0))
            return expanded
        if len(payload) != len(self._samples):
            raise ValueError(f"rows {len(payload)} do not match samples {len(self._samples)}")
        return [
            torch.full(sample.shifted_labels.shape, float(scalar), dtype=torch.float32)
            for sample, scalar in zip(self._samples, payload)
        ]

    def _build_sample(
        self,
        state: RolloutState,
        use_3d_position_ids: bool,
        teacher_adapter: DistillationTrainerAdapter,
    ) -> TrainSample:
        input_ids = cast(list[int], state.input_ids)
        labels = cast(list[int], state.labels)
        shifted_labels = labels[1:]

        input_ids_t = cast(torch.LongTensor, torch.tensor(input_ids[:-1], dtype=torch.int64).unsqueeze(0))
        shifted_labels_t = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)

        rollout_logprobs: torch.Tensor | None = None
        if state.logprobs is not None:
            rollout_logprobs = torch.tensor(state.logprobs[1:], dtype=torch.float32).unsqueeze(0)

        position_ids = state.position_ids
        if use_3d_position_ids and (position_ids is None or position_ids.ndim != 3):
            seq_len = input_ids_t.size(-1)
            text_position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, 1, -1)
            # Mixed batches use three-axis MRoPE position IDs. Repeat the normal text
            # positions on every axis so text samples have the same rank as the VLM
            # samples when their sequence contexts are packed together.
            position_ids = np.broadcast_to(text_position_ids, (3, 1, seq_len)).copy()
        seq_ctx = get_train_seq_ctx(input_ids_t, position_ids, cast(dict | None, state.mm_info))
        seq_ctx.rollout_routed_experts = state.routed_experts  # type: ignore[assignment] # n,layer*expert
        return TrainSample(
            seq_ctx=seq_ctx,
            shifted_labels=shifted_labels_t,
            rollout_logprobs=rollout_logprobs,
            teacher_targets=teacher_adapter.rollout_teacher_targets(state, shifted_labels=shifted_labels),
            reward_score=float(state.reward["score"])
            if state.reward is not None and "score" in state.reward
            else None,
        )

    def _padding_loss_input(self, key: str, padding_len: int) -> torch.Tensor:
        if key == "shifted_labels":
            return torch.full((1, padding_len), -100, dtype=torch.long)
        if key in ("advantages", "rollout_logprobs", "returns", "old_values"):
            return torch.zeros(1, padding_len, dtype=torch.float32)
        if key == "teacher_indices":
            return torch.full((1, padding_len), -1, dtype=torch.long)
        if key in ("teacher_logprobs", "target_token_ids"):
            distillation_config = self.distillation_config
            assert distillation_config is not None, f"Padding {key} requires distillation_config"
            if key == "teacher_logprobs" and distillation_config.loss_config.uses_sampled_token_targets:
                return torch.zeros(1, padding_len, dtype=torch.float32)
            top_k = distillation_config.loss_config.top_k
            assert top_k is not None, f"Padding {key} requires loss_config.top_k"
            if key == "teacher_logprobs":
                return torch.zeros(1, padding_len, top_k, dtype=torch.float32)
            return torch.zeros(1, padding_len, top_k, dtype=torch.long)
        raise ValueError(f"Unknown pack loss input key: {key}")

    def _padding_seq_ctx(self, padding_len: int) -> SequenceContext:
        # Reduce the attention calculation time by using multiple short sequence packs.
        pad_tokens = tuple(torch.zeros(1, 1024, dtype=torch.long) for _ in range(padding_len // 1024))
        if padding_len % 1024 > 0:
            pad_tokens = pad_tokens + (torch.zeros(1, padding_len % 1024, dtype=torch.long),)
        pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")  # type: ignore[arg-type]
        pad_seq_ctx.num_padding = padding_len
        if self.attrs["use_3d_position_ids"]:
            position_ids_list = []
            for pad_token in pad_tokens:
                position_ids = torch.arange(pad_token.size(-1)).view(1, 1, -1).expand(3, 1, -1)
                position_ids_list.append(position_ids)
            pad_seq_ctx.position_ids = torch.cat(position_ids_list, dim=-1)  # type: ignore[assignment]
        if self.attrs["has_routed_experts"]:
            language_cfg = _language_cfg(self.model_cfg)
            pad_seq_ctx.rollout_routed_experts = torch.randint(
                low=0, high=language_cfg.n_routed_experts, size=(padding_len, 1, 1)
            )
        return pad_seq_ctx


def iter_cpu_packs(
    batch: TrainBatch,
    indices: Sequence[Sequence[int]],
    resolve_refs,
) -> Iterator[FullPack]:
    """Materialize packs one at a time, resolving each pack's refs before
    yield.

    Args:
        batch (TrainBatch): Batch holding the CPU samples.
        indices (Sequence[Sequence[int]]): Local sample indices per pack.
        resolve_refs: Callback that materializes object refs on the pack's ``seq_ctx``.

    Yields:
        FullPack: The current pack.
    """
    for pack_indices in indices:
        pack = batch.materialize(tuple(pack_indices))
        resolve_refs(pack.full_data["seq_ctx"])
        yield pack


def get_train_seq_ctx(
    input_ids: torch.LongTensor,
    position_ids: np.ndarray | None = None,
    multimodal_train_info: dict | None = None,
) -> SequenceContext:
    """Build a CPU ``SequenceContext`` for one training sample.

    Args:
        input_ids (torch.LongTensor): Model input tokens with shape ``(1, seq_len)``.
        position_ids (np.ndarray | None): Optional position ids. A 3D array triggers
            the VLM MRoPE layout; a prompt-only position segment is extended to cover
            the whole input.
        multimodal_train_info (dict | None): Optional multimodal payload with
            ``pixel_values``, ``image_grid_thw`` and ``num_img_tokens``.

    Returns:
        SequenceContext: The CPU sequence context of the sample.
    """
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cpu")
    position_ids_t = _to_cpu_tensor(position_ids, dtype=torch.long)
    if position_ids_t is not None and len(position_ids_t.shape) == 3:
        # Match get_rope_index_3: response text continues from a single global
        # max(T, H, W), not per-axis maxima.
        num_extend = input_ids.size(-1) - position_ids_t.size(-1)
        if num_extend > 0:
            max_value = position_ids_t.amax()
            response_position_ids = (
                torch.arange(1, num_extend + 1, device=position_ids_t.device, dtype=position_ids_t.dtype)
                .view(1, 1, -1)
                .expand(3, 1, -1)
                + max_value
            )
            position_ids_t = torch.cat([position_ids_t, response_position_ids], dim=-1)
        seq_ctx.position_ids = position_ids_t  # type: ignore[assignment]
        assert position_ids_t.size(-1) == input_ids.size(-1)

    if multimodal_train_info:
        seq_ctx.pixel_values = multimodal_train_info.get("pixel_values")
        seq_ctx.image_grid_thw = _to_cpu_tensor(multimodal_train_info.get("image_grid_thw"), dtype=torch.long)
        num_img_tokens = multimodal_train_info.get("num_img_tokens")
        if num_img_tokens is not None:
            seq_ctx.num_img_tokens = [num_img_tokens]
    return seq_ctx


def resolve_mm_inputs(seq_ctx: SequenceContext) -> None:
    """Materialize multimodal pixel-value object refs into one CPU tensor.

    Args:
        seq_ctx (SequenceContext): Packed CPU context whose pixel refs are resolved.

    Returns:
        None
    """
    pixel_values = seq_ctx.pixel_values
    if pixel_values is None:
        return
    if isinstance(pixel_values, np.ndarray):
        raise NotImplementedError("The case where pixel_values is a numpy array is not implemented yet.")
    assert isinstance(pixel_values, list), f"pixel_values should be list of tensor, got {type(pixel_values)}"
    pixel_values = ray.get(list(pixel_values))
    pixel_values = [torch.as_tensor(pixel_value) for pixel_value in pixel_values]
    seq_ctx.pixel_values = torch.cat(pixel_values, dim=0)


def resolve_pack_refs(
    seq_ctx: SequenceContext,
    *,
    model_cfg: TransformerConfig | BaseComposeConfig,
    pack_max_length: int,
) -> None:
    """Materialize multimodal and routed-expert refs of one pack on CPU.

    Never frees refs: every stage re-resolves from the pristine cache and the
    trainer releases the batch's remote resources after training.

    Args:
        seq_ctx (SequenceContext): Packed CPU context whose refs are resolved.
        model_cfg (TransformerConfig | BaseComposeConfig): Model config.
        pack_max_length (int): Pack capacity for dummy routed-expert padding.

    Returns:
        None
    """
    resolve_mm_inputs(seq_ctx)

    rollout_routed_experts = seq_ctx.rollout_routed_experts
    if rollout_routed_experts is None:
        return
    language_cfg = _language_cfg(model_cfg)
    if not getattr(language_cfg, "n_routed_experts", 0):
        # Dense model with stale refs: drop them; the forward routes on its own.
        seq_ctx.rollout_routed_experts = None
        return
    storage_dtype = _rollout_routed_experts_storage_dtype(language_cfg.n_routed_experts)

    if not isinstance(rollout_routed_experts, list):
        # All-padding pack: rebuild dummy experts at the packed (l, e) shape.
        assert isinstance(rollout_routed_experts, torch.Tensor), (
            f"padding experts should be a dummy tensor, bug got {type(rollout_routed_experts)}"
        )
        seq_ctx.rollout_routed_experts = torch.randint(
            low=0,
            high=language_cfg.n_routed_experts,
            size=(pack_max_length, language_cfg.num_hidden_layers, language_cfg.num_experts_per_tok),
            dtype=storage_dtype,
        )
        return

    out_rollout_routed_expert = []
    for rollout_routed_expert in rollout_routed_experts:
        if isinstance(rollout_routed_expert, torch.Tensor):
            # Padding slot inside a mixed pack.
            out_rollout_routed_expert.append(
                torch.randint(
                    low=0,
                    high=language_cfg.n_routed_experts,
                    size=(
                        rollout_routed_expert.size(0),
                        language_cfg.num_hidden_layers,
                        language_cfg.num_experts_per_tok,
                    ),
                    dtype=storage_dtype,
                )
            )
            continue
        if isinstance(rollout_routed_expert, ray.ObjectRef):
            rollout_routed_expert = ray.get(rollout_routed_expert)
        elif isinstance(rollout_routed_expert, list):
            rollout_routed_expert = np.concatenate(
                [ray.get(rollout_routed_expert_ref) for rollout_routed_expert_ref in rollout_routed_expert],
                axis=0,
            )[1:, ...]
        else:
            raise ValueError(f"Invalid rollout_routed_expert_refs type: {type(rollout_routed_expert)}")
        rollout_routed_expert = _as_rollout_routed_experts_tensor(
            rollout_routed_expert,
            n_routed_experts=language_cfg.n_routed_experts,
        ).reshape(-1, language_cfg.num_hidden_layers, language_cfg.num_experts_per_tok)
        out_rollout_routed_expert.append(rollout_routed_expert)

    seq_ctx.rollout_routed_experts = torch.cat(out_rollout_routed_expert, dim=0)  # max_len,l,e
    assert seq_ctx.rollout_routed_experts.size(0) == seq_ctx.input_ids.size(1), (
        f"rollout_routed_experts.size(0) {seq_ctx.rollout_routed_experts.size(0)} "
        f"!= input_ids.size(1) {seq_ctx.input_ids.size(1)}"
    )


def _language_cfg(model_cfg: TransformerConfig | BaseComposeConfig):
    return model_cfg.text_config if isinstance(model_cfg, BaseComposeConfig) else model_cfg


def _to_cpu_tensor(value: np.ndarray | None, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    assert isinstance(value, np.ndarray), f"Expected np.ndarray, got {type(value)}"
    return torch.as_tensor(value, dtype=dtype, device="cpu")


def _rollout_routed_experts_storage_dtype(n_routed_experts: int) -> torch.dtype:
    """Choose a compact lossless dtype for stored routed-expert IDs."""
    if n_routed_experts <= _UINT16_CAPACITY:
        return torch.uint16
    return torch.long


def _as_rollout_routed_experts_tensor(
    rollout_routed_experts: np.ndarray,
    *,
    n_routed_experts: int,
) -> torch.Tensor:
    """Convert finalized routed-expert IDs to their CPU storage dtype."""
    storage_dtype = _rollout_routed_experts_storage_dtype(n_routed_experts)
    if storage_dtype == torch.uint16 and rollout_routed_experts.dtype != np.uint16:
        if rollout_routed_experts.size > 0:
            min_expert_id = rollout_routed_experts.min()
            max_expert_id = rollout_routed_experts.max()
            if min_expert_id < 0 or max_expert_id >= _UINT16_CAPACITY:
                raise ValueError(
                    f"Routed-expert IDs cannot be represented as uint16: min={min_expert_id}, max={max_expert_id}"
                )
    return torch.as_tensor(rollout_routed_experts, dtype=storage_dtype)
