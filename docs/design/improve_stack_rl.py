"""四层 stack 整体方案的设计伪代码；配套 improve_stack_rl.md。

不用于运行训练：省略/迁移的 Implementation 用 ... 或 EXISTING_* 标出。
EXISTING_* 是说明性占位，不保证是现有符号，也不是建议新增一批转发 helper。
本文件只要求 Python 语法有效，不承诺 import、类型检查或运行通过。

S1 = #2102 / d74e4be8：canonical train data
S2 = #2185 / 0323086f：index plan / worker-local TrainBatch
S3 = #2188 / 542110e0：head / loss build
S4 = #2186 / d42ce89f：PPO targets / roles / update lifecycle

按目标 Module 组织；同一 Module 跨层扩展时在方法上标注。
FullPack / TrainMeta / ActorSnapshot 等是数据记录，不是新的管理层。
仅展开有设计意义的变更；未展开的初始化、日志、配置和 @ray_method 沿用既有实现。
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Iterator, Literal

from pydantic import BaseModel, ConfigDict


if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from xtuner.v1.rl.distillation.train_teacher_manager import TrainTeacherTimings
    from xtuner.v1.rl.trainer.pack import TokenBatchProto


# ============================================================================
# A. [S1] xtuner/v1/data_proto/rl_data.py + AgentLoop / produce_utils 调用方
# ============================================================================


@dataclass(frozen=True)
class TrainMeta:
    # num_tokens 保留在 RolloutState 的既有字段中，不再复制到 metadata。
    prompt_length: int
    response_length: int
    supervised_tokens: int                 # count(full labels[1:] != -100)
    position_layout: Literal["1d", "mrope_3d"]
    has_rollout_logprobs: bool
    has_routed_experts: bool
    segment_index: int | None               # 迁移已有 agent_trace_segment_index


class RolloutState(BaseModel):
    """[S1] 深化既有 BaseModel；只展开训练契约，其他字段沿用现有声明。

    普通成员方法在接收 Ray 数据的进程本地执行，不是 Ray actor RPC。
    train_meta 是显式序列化字段；不使用 PrivateAttr、property 替换或整体 frozen。
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    input_ids: list[int] | None = None
    labels: list[int] | None = None
    logprobs: list[float] | None = None
    num_tokens: int | None = None
    train_meta: TrainMeta | None = None
    # prompt_ids / response_ids / position_ids / routed_experts / status 等保持现有声明。

    def set_training_fields(
        self, *, input_ids: list[int], labels: list[int], logprobs: list[float] | None
    ) -> None:
        """[S1] 验证并发布完整未 shift 字段、训练长度和 metadata。"""
        # 复制轻量 token lists，避免调用方保留的 list 别名随后修改已发布字段。
        ids, target_labels = list(input_ids), list(labels)
        full_logprobs = list(logprobs) if logprobs is not None else None
        meta = self._build_train_meta(input_ids=ids, labels=target_labels, logprobs=full_logprobs)
        # 验证成功后才赋值；同步方法不插入 await，也不声称提供跨线程事务。
        self.input_ids = ids
        self.labels = target_labels
        self.logprobs = full_logprobs
        self.num_tokens = len(ids) - 1
        self.train_meta = meta

    def apply_response_mask(self, effective_mask: list[bool]) -> None:
        """[S1] 消费时更新 labels 与监督计数；不改长度、prompt 或原有语义洞。"""
        if self.labels is None or self.train_meta is None:
            raise ValueError("Training fields must be published before applying a response mask")
        if len(effective_mask) != len(self.response_ids or []):
            raise ValueError("Response mask must align with response_ids")
        labels = list(self.labels)
        start = len(labels) - len(effective_mask)
        for index, keep in enumerate(effective_mask):
            if not keep:
                labels[start + index] = -100
        meta = replace(self.train_meta, supervised_tokens=sum(label != -100 for label in labels[1:]))
        self.labels = labels
        self.train_meta = meta
        # 不重新推导 prompt_length：缺失显式 prompt 时，mask 后的非监督数会变大。

    def reset_response(self) -> None:
        """[S1] 重生成前失效响应及其训练产物；重试与过期策略留在调用方。"""
        ...  # 迁入现有 reset 的 routed refs 释放段落，不扩展释放范围或共享 refs 所有权。
        self.tokens = list(self.prompt_ids) if self.prompt_ids is not None else None
        self.response = ""
        self.response_ids = []
        self.logprobs = []
        self.teacher_targets = None
        self.routed_experts = None
        self.finish_reason = None
        self.response_model_steps = []
        self.reward = None
        self.error_msg = None
        self.input_ids = None
        self.labels = None
        self.num_tokens = None
        self.train_meta = None
        # 保留 prompt / multimodal 上下文；不在此选择 status 或改 partial-rollout 策略。

    def validate_training(self) -> None:
        """[S1] 只读检查本样本训练字段与摘要，不修补数据或改变 group 状态。"""
        if self.input_ids is None or self.labels is None or self.train_meta is None:
            raise ValueError("Training fields and train_meta must be present")
        expected = self._build_train_meta(
            input_ids=self.input_ids, labels=self.labels, logprobs=self.logprobs,
            prompt_length=self.train_meta.prompt_length,
        )
        if self.num_tokens != len(self.input_ids) - 1 or self.train_meta != expected:
            raise ValueError("Training fields and metadata are inconsistent")
        if len(self.response_model_steps or []) != len(self.response_ids or []):
            raise ValueError("response_model_steps must align with response_ids")
        ...  # 其余既有 position/teacher/routing 的单样本验证在此保留。

    def _build_train_meta(
        self, *, input_ids: list[int], labels: list[int], logprobs: list[float] | None,
        prompt_length: int | None = None,
    ) -> TrainMeta:
        if len(input_ids) < 2 or len(input_ids) != len(labels):
            raise ValueError("Training input_ids and labels must be full and aligned")
        if logprobs is not None and len(logprobs) != len(input_ids):
            raise ValueError("Full rollout logprobs must align with full input_ids")
        response_length = len(self.response_ids or [])
        if response_length > len(labels):
            raise ValueError("Response suffix exceeds full labels")
        kept = sum(label != -100 for label in labels[1:])
        if prompt_length is None:
            prompt_ids = self.extra_fields.get("train_prompt_ids") or self.prompt_ids
            prompt_length = len(prompt_ids) if prompt_ids else len(input_ids) - 1 - kept
        if prompt_length < 0:
            raise ValueError("prompt_length must be nonnegative")
        return TrainMeta(
            prompt_length=prompt_length,
            response_length=response_length,
            supervised_tokens=kept,
            position_layout=(
                "mrope_3d" if self.position_ids is not None and self.position_ids.ndim == 3 else "1d"
            ),
            has_rollout_logprobs=logprobs is not None,
            has_routed_experts=self.routed_experts is not None,
            segment_index=self.extra_fields.get("agent_trace_segment_index"),
        )


def validate_train_group(group: list[RolloutState], *, task_name: str) -> None:
    """[S1] group 入口只保留组级条件，单样本一致性交给 RolloutState。"""
    if not group or any(not isinstance(state, RolloutState) or state.status != Status.COMPLETED for state in group):
        raise ValueError(f"task={task_name!r}: training requires a nonempty COMPLETED rollout group")
    for state in group:
        state.validate_training()
    # 沿用现有 task/group/rollout 错误归因；入口决定丢弃整个 group。


class AgentLoop:
    def canonicalize_train_fields(self, group: list[RolloutState]) -> list[RolloutState]:
        """[S1] 仅展开既有类的集成点；tokenizer、trace 和任务语义仍由 AgentLoop 组装。"""
        for state in group:
            if state.status != Status.COMPLETED:
                continue
            try:
                # 普通响应组装 full ids/labels/logprobs；自定义 agent 复用已组装的 full 字段。
                # EXISTING_* 包含原有 response tokenization、prompt 前缀与可选 logprobs 对齐。
                ids, labels, full_logprobs = EXISTING_ASSEMBLE_FULL_TRAIN_FIELDS(self, state)
                state.set_training_fields(input_ids=ids, labels=labels, logprobs=full_logprobs)
            except Exception as exc:
                self.logger.error("Canonicalize failed: %s", exc)
                state.status = Status.FAILED
                state.error_msg = f"canonicalize_train_fields failed: {exc}"
        return group


async def take_train_batch(*, task_runners, replay_buffer, current_train_step, **kwargs):
    """[S1] 仅展开既有消费入口；mask 的策略计算与实际状态更新分开。"""
    batch_by_task, consumed_counts = await replay_buffer.take_batch(kwargs["task_batch_sizes"])
    for task in task_runners:
        for group in batch_by_task.get(task.task_name, []):
            masks = calculate_group_effective_response_masks(
                group, current_train_step=current_train_step, token_stale_threshold=task.token_stale_threshold
            )
            for state, mask in zip(group, masks):
                if mask is not None:
                    state.apply_response_mask(mask)
    # 任务归因、consumed/leftover 与 ProduceBatchResult 汇总保持现有实现。
    return EXISTING_BUILD_PRODUCE_RESULT(batch_by_task, consumed_counts, **kwargs)


# replay_buffer / rollout worker 的原 reset 调用统一改为 state.reset_response()。
# 不保留 finalize_train_fields(state) / apply_response_mask(state, mask) 等外部转发入口。
# Ray 传输返回更新后的 RolloutState；普通方法不会自动更新发送方或已 put 的快照。


# ============================================================================
# B. [S2] trainer/pack.py + trainer/train_batch.py
# ============================================================================


@dataclass(frozen=True)
class RankPlan:
    global_indices: tuple[int, ...]         # 本地 sample index → 全局 sample index
    steps: tuple                           # step → pack → local sample indices


@dataclass(frozen=True)
class PackPlan:
    ranks: tuple[RankPlan, ...]
    metrics: dict[str, float]

    def restore_rows(self, rows_by_dp: list[TokenBatchProto]) -> list[Tensor]:
        """[S2] 纯索引逆映射；每 DP 恰好一份，不能传所有 replica 的重复行。"""
        if len(rows_by_dp) != len(self.ranks):
            raise ValueError("Expected exactly one payload per DP rank")
        size = sum(len(rank.global_indices) for rank in self.ranks)
        result = [None] * size
        for rank, payload in zip(self.ranks, rows_by_dp):
            rows = payload.chunk()
            if len(rows) != len(rank.global_indices):
                raise ValueError("Transport rows do not match RankPlan")
            for global_id, row in zip(rank.global_indices, rows):
                if result[global_id] is not None:
                    raise ValueError("Duplicate sample in plan")
                result[global_id] = row
        if any(row is None for row in result):
            raise ValueError("Incomplete sample coverage")
        return result


class RLDataPacker:
    def plan(self, lengths: list[int]) -> PackPlan:
        """[S2] 深化原 pack：算法、local 重编号、统计一起封装。

        原生/greedy/balance/LPT 保留既有行为；LPT 从 S4 移到 S2。
        同一 update 各 DP 的 pack 数对称，空 pack 用全 padding 表达。
        """
        global_steps, padding_count = EXISTING_PACK_ALGORITHM(lengths, self.config)
        ranks = []
        for steps in global_steps:
            global_ids = tuple(sorted({i for step in steps for pack in step for i in pack}))
            to_local = {global_id: local_id for local_id, global_id in enumerate(global_ids)}
            ranks.append(RankPlan(
                global_indices=global_ids,
                steps=tuple(tuple(tuple(to_local[i] for i in pack) for pack in step) for step in steps),
            ))
        return PackPlan(tuple(ranks), EXISTING_PACK_METRICS(lengths, global_steps, padding_count))


@dataclass(frozen=True)
class FullPack:
    indices: tuple[int, ...]
    lengths: tuple[int, ...]                # 包括 prompt / observation，不仅是 action
    full_data: dict                         # 完整 CPU seq_ctx + full token fields


class TrainBatch:
    """[S2] 真正取代 worker 的三个缓存成员；S4 扩展目标方法。

    _samples 是唯一完整 CPU 源，外部不直接修改其字段。
    teacher Adapter 使用已有 distillation Module，不反向调用 actor 对象。
    无 Ray handles / model engine / loss context；源 seq 的 refs 只借用。
    """

    def __init__(self, states, rank_plan: RankPlan, attrs, teacher_adapter):
        self.plan = rank_plan
        self.attrs = attrs
        self._samples = []
        for state in states:
            # [S2] 完整 token → next-token prediction 坐标，仅此一次。
            labels = torch.tensor(state.labels[1:], dtype=torch.long)[None, :]
            seq = EXISTING_GET_TRAIN_SEQ_CTX(
                input_ids=torch.tensor(state.input_ids[:-1])[None, :],
                position_ids=state.position_ids,
                mm_info=state.mm_info,
                use_3d_position_ids=attrs.use_3d_position_ids,
            )
            seq.rollout_routed_experts = state.routed_experts
            fields = {
                "shifted_labels": labels,
                "rollout_logprobs": (
                    torch.tensor(state.logprobs[1:], dtype=torch.float32)[None, :]
                    if state.logprobs is not None else None
                ),
                **(teacher_adapter.get_rollout_targets(state, shifted_labels=labels) if teacher_adapter else {}),
            }
            self._samples.append(EXISTING_SAMPLE_RECORD(seq, fields, reward=state.reward))
        # validate_train_group、presence/schema、超长样本等验证在入口做一次。
        # S2 无 critic 依赖；S4 添加的 old_values / returns 初始为空。

    def materialize(self, indices: tuple[int, ...]) -> FullPack:
        """[S2] 所有字段使用同一布局；含空 pack 的有限字段 padding 表留在这里。"""
        samples = [self._samples[i] for i in indices]
        lengths = tuple(sample.fields["shifted_labels"].shape[1] for sample in samples)
        pad = self.attrs.pack_max_length - sum(lengths)
        if pad < 0:
            raise ValueError("Pack exceeds capacity")
        # 从 worker.pack_seq_ctx / cat_pack_fields 迁移，不另造通用反射框架。
        # 保留 MRoPE global-amax 续写、短 dummy sequences 和 routed padding。
        seq_ctx, fields = EXISTING_CPU_CONCAT_AND_PAD(samples, pad, self.attrs)
        return FullPack(indices, lengths, {"seq_ctx": seq_ctx, **fields})

    def unpack_rows(self, pack: FullPack, full_rows: Tensor) -> list[tuple[int, Tensor]]:
        """[S2] 调用方已去 SP padding；这里去 pack padding，恢复样本行。"""
        rows = full_rows[:, :sum(pack.lengths)].detach().cpu()
        output, offset = [], 0
        for sample_id, length in zip(pack.indices, pack.lengths):
            output.append((sample_id, rows[:, offset:offset + length]))
            offset += length
        return output

    def set_advantages(self, payload: list[float] | TokenBatchProto) -> None:
        """[S2] scalar 广播或 token rows；先校验全批再写，统一无效位置为 0。"""
        rows = EXISTING_EXPAND_AND_VALIDATE_ROWS(payload, self._samples)
        for sample, row in zip(self._samples, rows):
            sample.fields["advantages"] = row.masked_fill(sample.fields["shifted_labels"].eq(-100), 0)

    def set_old_values(self, rows: list[Tensor]) -> None:
        """[S4 扩展] 全部样本的完整冻结值；不允许 fit 内临时补前向。"""
        EXISTING_VALIDATE_FULL_ROWS(rows, self._samples)
        for sample, row in zip(self._samples, rows):
            sample.fields["old_values"] = row.detach().float().cpu()

    def set_ppo_targets(self, advantages: TokenBatchProto, returns: TokenBatchProto) -> None:
        """[S4 扩展] 两份全量校验成功后再安装，returns 不掩码/归一化。"""
        adv, ret = advantages.chunk(), returns.chunk()
        EXISTING_VALIDATE_FULL_ROWS(adv, self._samples)
        EXISTING_VALIDATE_FULL_ROWS(ret, self._samples)
        for sample, a, r in zip(self._samples, adv, ret):
            labels = sample.fields["shifted_labels"]
            sample.fields["advantages"] = a.reshape_as(labels).masked_fill(labels.eq(-100), 0)
            sample.fields["returns"] = r.reshape_as(labels)

    def trajectory_data(self):
        """[S4 扩展] 给数值 Module 的 full values/scores/masks/cu_seq_lens。

        只读完整 CPU 数据，隐藏 _samples；不包含 pack padding，不含 Ray。
        返回 keys: values/scores/mask/cu_seq_lens，与 estimator 参数对应。
        无本地样本时返回空 tensors、cu_seq_lens=[0]，仍允许参与全局归约。
        """
        ...

    def require_targets(self, role: Literal["actor", "critic"]) -> None:
        """[S2 actor；S4 扩展 critic] 完整批次的前置条件，不能由 role 读裸 sample 验证。"""
        fields = ("advantages",) if role == "actor" else ("old_values", "returns")
        for sample in self._samples:
            for name in fields:
                if sample.fields.get(name) is None:
                    raise ValueError(f"{role} requires complete {name} rows")
        # 空 DP 无实际 sample：允许通过，后续仍执行 padding packs 与 collective。

    def close(self) -> None:
        """[S2] 仅关闭本地缓存；远端 ObjectRef 由 RLTrainer 在整批消费后释放。"""
        self._samples.clear()


# [S2] 同步逐 pack 物化位于上述 Module，供 S4 RoleWorker 使用。
# 它是私有 Implementation，不是一个供外部挑选 Adapter 的框架。
def iter_cpu_packs(batch: TrainBatch, indices, resolve_refs) -> Iterator[FullPack]:
    """[S2] 调用方每取一项，才同步物化并解析当前 pack 的 CPU 数据。

    ray.get 沿用有限 timeout；异常直接沿当前调用栈传播。
    调用方随后执行 H2D / SP split / forward，不提前构造整批 CPU packs。
    optimizer-step / whole-batch teacher 所需的聚合仍由各调用方保留。
    """
    for pack_indices in indices:
        pack = batch.materialize(pack_indices)
        resolve_refs(pack.full_data["seq_ctx"])
        yield pack


# ============================================================================
# C. [S3] loss/* + rl/loss/* + model existing build_loss_ctx_batch
# ============================================================================


class CriticLossKwargs:
    """[S3] 展示要补齐的既有 Kwargs 方法；原数据字段/构造器省略。"""
    def sp_split(self, sp_mesh, *, ignore_idx=-100):
        self.shifted_labels = sp_split(self.shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=ignore_idx)
        for name in ("returns", "old_values", "loss_weight"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, sp_split(value, sp_mesh=sp_mesh, split_dim=1, padding_value=0))
        return self

    def to(self, device):
        for name in ("shifted_labels", "returns", "old_values", "loss_weight"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device))
        return self


# [S3] CELossKwargs 对 labels / loss_weight 做相同 split/to；LogProb 可复用它。
# 这是本次必需改动，不能直接依赖当前只处理 labels 的 CELossKwargs。


class CriticLossConfig:
    def build(self, data, sp_mesh=None, *, device=None):
        """[S3] 完整监督字段进来，本 rank context 出去；此处无 collective。"""
        labels = data["shifted_labels"]
        returns, old_values = data["returns"], data.get("old_values")
        # 归一化仅末尾 channel，不 squeeze batch/token 维。
        if returns.ndim == 3 and returns.shape[-1] == 1:
            returns = returns.squeeze(-1)
        if old_values is not None and old_values.ndim == 3 and old_values.shape[-1] == 1:
            old_values = old_values.squeeze(-1)
        if returns.shape != labels.shape or (old_values is not None and old_values.shape != labels.shape):
            raise ValueError("Critic target shape must match [B,L] labels")
        device = DEVICE if device is None else device
        kwargs = CriticLossKwargs(
            shifted_labels=labels,
            returns=returns,
            old_values=old_values,          # None 保留 MSE；PPO role 要求非空
            loss_weight=labels.ne(self.ignore_idx).float(),
        )
        if sp_mesh is not None and sp_mesh.size() > 1:
            kwargs.sp_split(sp_mesh, ignore_idx=self.ignore_idx)
        kwargs = kwargs.to(device)
        return CriticLossContext(self, kwargs)


class CELossConfig:
    def build(self, data, sp_mesh=None, *, device=None):
        """[S3] sample/square 权重在完整样本上生成；build_batches 不再 gather 标签。"""
        device = DEVICE if device is None else device
        labels = data["shifted_labels"]
        weights = EXISTING_FULL_SAMPLE_WEIGHTS(labels, data["seq_ctx"], self.loss_reduction, self.ignore_idx)
        kwargs = CELossKwargs(shifted_labels=labels, loss_weight=weights)
        if sp_mesh is not None and sp_mesh.size() > 1:
            kwargs.sp_split(sp_mesh, ignore_idx=self.ignore_idx)
        return CELossContext(self, kwargs.to(device))


class MTPLossConfig:
    def build(self, data, sp_mesh=None, *, device=None):
        """[S3] 先逐样本 roll，再复用 CE 的 full→local 准备 Implementation。"""
        labels = roll_packed_tensor(
            data["shifted_labels"], data["seq_ctx"].cu_seq_lens_k,
            shifts=-self.mtp_depth, dim=1, fill_value=self.ignore_idx,
        )
        # rollout logprobs 也按相同边界 roll；CE/KL/TV 保持各自数学及有效 mode。
        ...


class CriticLossContext:
    @classmethod
    def build_batches(cls, contexts):
        """[S3] 一个 update 的所有 microbatch；每 context 仅调用一次。"""
        denominator = sum(ctx.loss_kwargs.loss_weight.sum() for ctx in contexts)
        if dist.is_initialized():            # 保留非分布式 SFT/loss 的公开支持
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        denominator = denominator.clamp_min(1)
        for ctx in contexts:
            ctx._batch_size = len(contexts)
            ctx.loss_kwargs.loss_weight = ctx.loss_kwargs.loss_weight / denominator
        return contexts
        # 示意 critic/token 权重；不能把此公式套到 Oreal 或 auxiliary losses。
        # 沿用既有 WORLD SUM + engine/FSDP 梯度校准，不再除 SP / microbatch 数。


# [S3] RL/Distillation/LogProb builders 同样接受 full data + mesh + device。
# full old/ref 字段若在 build 输入中存在，则切分；late local fields 只在 build 后绑定。
# IS 以 masked_fill 新建 labels，不原地修改完整源数据。
# model.build_loss_ctx_batch 隐藏主 loss/MTP/aux 组合；SFT override 显式传入。


class BaseModel:
    def from_hf(self, hf_path, strict=True):
        """[S3] 收拢现有 head 加载判断，不新增另一套 key mapping。"""
        self._init_load_spec()               # 依据当前 runtime layout 刷新完整映射
        self._assert_load_spec_initialized()
        self._hf_path = Path(hf_path)
        self._validate_head_checkpoint(self._hf_path)  # 保留既有 critic→actor 拒绝规则
        loader = HFCheckpointLoader(hf_path)
        # 复用既有 _head_load_exceptions；严格检查和初始化共享这一次全局判断。
        # 不新增 _resolve_head_transition 或 HeadPolicy / transition 数据记录。
        head_exceptions = self._head_load_exceptions(loader)
        result = self._load_params(loader, strict=strict, head_exceptions=head_exceptions)
        return self._init_missing_critic_head(head_exceptions[0], *result)

    def _head_load_exceptions(self, loader) -> tuple[set[str], set[str]]:
        # 现有 _critic_head_hf_keys 从 load_spec_mapping["lm_head.weight"].global_hf_keys 取值；
        # 已应用模型族映射及 hf_key_mapping，不能硬编码为 {"value_head.weight"}。
        value_head_keys = self._critic_head_hf_keys()
        if value_head_keys is None:
            return set(), set()             # 非 critic：没有 head 缺失例外
        present = set(loader.weight_map)    # checkpoint 全局 keys，不是本 rank loaded_keys
        if value_head_keys <= present:
            return set(), set()             # 完整 critic head：加载已有权重，不初始化
        if value_head_keys & present:
            raise RuntimeError("Checkpoint contains only part of the value head")
        # value head 完全缺失：仅放行它的缺失，以及 actor 词表 head 的多余 keys。
        # 沿用现有模型族映射获得 actor-side keys；骨干和 shape 检查不放宽。
        return value_head_keys, set(self._model_hf_keys("lm_head.weight"))

    def _load_params(self, loader, *, strict, head_exceptions):
        allowed_missing, allowed_extra = head_exceptions
        expected = {key for spec in self.load_spec_mapping.values() for key in spec.global_hf_keys}
        present = set(loader.weight_map)
        if strict:
            missing = expected - present - allowed_missing
            unexpected = present - expected - allowed_extra
            if missing or unexpected:
                raise RuntimeError(f"Missing checkpoint keys: {missing}; unexpected keys: {unexpected}")
        # 此处接既有 _load_params 的参数读取、shape 校验及本地 loaded/unloaded/missing 汇总；
        # 末尾严格检查继续仅扣除同一 allowed_missing，不再调用 _head_load_exceptions。
        ...

    def _init_missing_critic_head(self, missing_value_head_keys, loaded, unloaded, missing):
        # 原方法改为接收上面的全局决策；不再重新查 weight_map 或用 rank-local loaded 判断。
        if missing_value_head_keys:
            # 既有初始化数学与 DTensor 行为保持，所有 rank 都进入同一初始化路径。
            init_params(self.lm_head.weight, partial(
                torch.nn.init.normal_, mean=0.0, std=1.0 / (self.config.hidden_size + 1),
            ))
            loaded.add("lm_head.weight")    # runtime 名仍是 lm_head；HF 名来自映射
            unloaded.discard("lm_head.weight")
            missing.difference_update(missing_value_head_keys)
        return loaded, unloaded, missing

    # 既有 HeadConfig.build / build_head / head_fp32 / to_hf_key_list 保留。
    # critic→actor 拒绝；critic→critic 不重置已加载 head；HF plan 与 DCP 不合并。


# ============================================================================
# D. [S4] advantage/base.py / gae.py：数值 Seam，不依赖 Ray / worker
# ============================================================================


class AdvantageEstimator(ABC):
    """既有 scalar Interface；GRPO / RLOO / OPO / PassK Adapter 保持。"""
    @abstractmethod
    def compute(self, rewards, group): ...


class TokenAdvantageEstimator(ABC):
    """[S4] 与 scalar 平行，不能继承后令 scalar compute 抛错。"""
    reward_scope: Literal["segment", "session"]
    normalize_actor_advantage: bool

    @abstractmethod
    def compute_segment_gae(self, values, scores, mask, cu_seq_lens):
        """返回未归一化 adv/returns；现有末 action reward / 零 bootstrap 语义。"""
        ...


class GAEEstimator(TokenAdvantageEstimator):
    def compute_segment_gae(self, values, scores, mask, cu_seq_lens):
        frozen = values.detach().float()
        rewards = terminal_rewards(scores, mask, cu_seq_lens).to(frozen.device)
        adv = action_gae(frozen, rewards, mask, cu_seq_lens, self.gae_gamma, self.gae_lambda)
        returns = torch.where(mask.to(frozen.device).bool(), adv + frozen, 0.0)
        return adv.detach(), returns.detach()


class DecoupledGAEEstimator(TokenAdvantageEstimator):
    reward_scope = "segment"                # 本次不自动开放 session

    def compute_segment_gae(self, values, scores, mask, cu_seq_lens):
        ...                                # 现有独立 actor/critic 折扣实现保持


def session_token_targets(estimator, states, full_values):
    """[S4] 迁移原 compute_session_gae 的完整拼/算/拆逻辑；无 RPC。

    session_id → rollout_id → sample index 的既有 fallback；按 segment_index 排序。
    每 session 最后一个有 score 的 segment 提供 reward；拼完整 values/masks；
    compute_segment_gae 后按原宽度拆回原 sample 顺序，再一次全局 actor 归一化。
    非 action 置零；returns 保持原数值。不得按 packing 顺序拼轨迹。
    """
    if isinstance(estimator, DecoupledGAEEstimator):
        raise ValueError("Decoupled GAE remains segment-only")
    masks = [torch.tensor(s.labels[1:]).ne(-100) for s in states]
    groups = {}
    for index, state in enumerate(states):
        key = (("session", state.session_id) if state.session_id is not None else
               ("rollout", state.rollout_id) if state.rollout_id is not None else ("index", index))
        groups.setdefault(key, []).append(index)
    advantages, returns = [None] * len(states), [None] * len(states)
    for members in groups.values():
        ordered = sorted(members, key=lambda i: (
            states[i].train_meta.segment_index if states[i].train_meta.segment_index is not None else i, i
        ))
        widths = [full_values[i].numel() for i in ordered]
        values = torch.cat([full_values[i].flatten().float().cpu() for i in ordered])
        mask = torch.cat([masks[i] for i in ordered])
        reward = next((float(states[i].reward["score"]) for i in reversed(ordered)
                       if states[i].reward is not None and "score" in states[i].reward), 0.0)
        adv, ret = estimator.compute_segment_gae(
            values, torch.tensor([reward]), mask, torch.tensor([0, values.numel()], dtype=torch.int32)
        )
        for i, a, r in zip(ordered, adv.split(widths), ret.split(widths)):
            advantages[i], returns[i] = a, r
    if estimator.normalize_actor_advantage:
        valid = torch.cat([a[m] for a, m in zip(advantages, masks)]).double()
        moments = torch.stack((valid.sum(), valid.square().sum(), valid.new_tensor(valid.numel())))
        advantages = normalize_with_moments(advantages, masks, moments)
    # 这里返回 CPU Tensor rows，Controller 才封装 TokenBatchProto；不返回 list[float]。
    return [a.masked_fill(~m, 0) for a, m in zip(advantages, masks)], returns


# ============================================================================
# E. [S4] actor_worker.py / critic_worker.py / role_worker.py
# ============================================================================


@dataclass(frozen=True)
class ActorSnapshot:
    """[S4] 单个 pack 的冻结 loss 输入，所有张量为 local CPU。

    fit 的局部 snapshots_by_pack 按 pack.indices 索引；只在当前 worker / batch / SP 布局使用。
    """
    shifted_labels: Tensor
    old_logprobs: Tensor
    is_weights: Tensor | None
    ref_logprobs: Tensor | None
    teacher_logprobs: Tensor | None
    target_token_ids: Tensor | None


class BaseRLLossContext:
    def bind_actor_snapshot(self, snapshot: ActorSnapshot) -> None:
        """[S4] 具名接收完整阶段产物；扩展 S3 loss Module，不再 split。"""
        kw = self.loss_kwargs
        if snapshot.shifted_labels.shape != kw.shifted_labels.shape:
            raise ValueError("Snapshot belongs to a different local layout")
        kw.shifted_labels = snapshot.shifted_labels.to(kw.shifted_labels.device)
        kw.old_logprobs = snapshot.old_logprobs.to(kw.shifted_labels.device)
        for field in ("is_weights", "ref_logprobs", "teacher_logprobs", "target_token_ids"):
            value = getattr(snapshot, field)
            if value is not None:
                setattr(kw, field, value.to(kw.shifted_labels.device))
        # 构建训练 weights/denominator 必须在此之后，IS 最终 mask 才会生效。


# 以下两个目标文件 import role_worker.RoleWorker（本稿 H 节），不是新建 Ray actors。
class ActorWorker(RoleWorker):
    role_name = "actor"

    def fit(self, batch: TrainBatch):
        """[S4] 显式依赖 batch；不再 get_cached_batch / host.pack_*。"""
        batch.require_targets("actor")
        # 一次 fit 的临时产物用局部变量表达，不放进长期存在的 worker 成员。
        snapshots_by_pack = self._collect_old_policy_snapshots(batch)
        try:
            # 原 _fit_stages 留在 ActorWorker 内，不建立通用 pipeline。
            self._stage_ref_logprobs(batch, snapshots_by_pack)
            self._stage_teacher_outputs(batch, snapshots_by_pack)
            return self._fit_policy_then_sft(batch, snapshots_by_pack)
        finally:
            # 显式结束本批冻结缓存，即使异常 traceback 仍持有该字典也不保留整批 tensors。
            snapshots_by_pack.clear()      # Controller 关闭本地 batch；RLTrainer 释放远端 refs

    def _collect_old_policy_snapshots(self, batch: TrainBatch) -> dict[tuple[int, ...], ActorSnapshot]:
        """[S4] 展开最关键的保存顺序；metrics/FP8 预热保留既有 Implementation。"""
        snapshots_by_pack: dict[tuple[int, ...], ActorSnapshot] = {}
        indices = [pack for step in batch.plan.steps for pack in step]
        for pack in iter_cpu_packs(batch, indices, self._resolve_refs):
            data = pack.full_data
            seq = data["seq_ctx"].copy().to(self.device)
            if self.sp_mesh.size() > 1:
                seq = seq.split(self.sp_mesh)
            ctx = self.loss_cfg.build(data, sp_mesh=self.sp_mesh, device=self.device)
            read_ctx = self.logprob_cfg.build(data, sp_mesh=self.sp_mesh, device=self.device)
            ctx.loss_kwargs.old_logprobs = self._forward_logprobs(seq, read_ctx.loss_kwargs.shifted_labels)
            if ctx.loss_kwargs.rollout_logprobs is not None:
                ctx.compute_rollout_is(self.sp_mesh, seq.seq_lens_q)
            # IS 已更新的 labels/weights 与 old logprobs 一起保存；不能只保存预测。
            snapshots_by_pack[pack.indices] = self._capture_snapshot(ctx)
        return snapshots_by_pack

    def _stage_ref_logprobs(
        self,
        batch: TrainBatch,
        snapshots_by_pack: dict[tuple[int, ...], ActorSnapshot],
    ) -> None:
        if not self._has_ref:
            return
        indices = [pack for step in batch.plan.steps for pack in step]
        self._ref_model.to_device(self.device)
        try:
            for pack in iter_cpu_packs(batch, indices, self._resolve_refs):
                data = pack.full_data
                seq = data["seq_ctx"].copy().to(self.device)
                if self.sp_mesh.size() > 1:
                    seq = seq.split(self.sp_mesh)
                ctx = self.loss_cfg.build(data, sp_mesh=self.sp_mesh, device=self.device)
                ctx.bind_actor_snapshot(snapshots_by_pack[pack.indices])
                ctx.loss_kwargs.ref_logprobs = self._forward_ref_logprobs(seq, ctx.loss_kwargs.shifted_labels)
                snapshots_by_pack[pack.indices] = self._capture_snapshot(ctx)
                ...                         # 保留既有 KL 指标累计；此处不校准训练 loss
        finally:
            self._ref_model.to_device("cpu")  # reference 每阶段一次上下卡，不调用 host
        ...                                 # 保留既有 KL 全局归约与日志

    def _stage_teacher_outputs(
        self,
        batch: TrainBatch,
        snapshots_by_pack: dict[tuple[int, ...], ActorSnapshot],
    ) -> TrainTeacherTimings | None:
        """[S4] actor 临时让出设备；teacher 整批计算完成后恢复 actor。"""
        if self._train_teacher_manager is None:
            return None

        # 调用 RoleWorker 的继承方法，只搬运 self.engine；critic 始终保持宿主安排的状态。
        self.offload_optimizer()
        self.offload_model()
        DEVICE_MODULE.empty_cache()

        indices = [pack for step in batch.plan.steps for pack in step]
        seq_ctx_list, loss_ctx_list = [], []
        try:
            for pack in iter_cpu_packs(batch, indices, self._resolve_refs):
                data = pack.full_data
                seq = data["seq_ctx"].copy().to(self.device)
                if self.sp_mesh.size() > 1:
                    seq = seq.split(self.sp_mesh)
                ctx = self.loss_cfg.build(data, sp_mesh=self.sp_mesh, device=self.device)
                ctx.bind_actor_snapshot(snapshots_by_pack[pack.indices])
                seq_ctx_list.append(seq)
                loss_ctx_list.append(ctx)

            # 保留 teacher 的 whole-batch Interface，不在 pack 循环中反复上下卡。
            # manager 自己在 finally 中 offload_all_to_cpu；这里不重复接管 teacher 生命周期。
            timings = self._train_teacher_manager.compute_teacher_outputs(seq_ctx_list, loss_ctx_list)
            for pack_indices, ctx in zip(indices, loss_ctx_list):
                # teacher 将结果写入 context；保存 old/ref/IS 与新增 teacher 字段的完整快照。
                snapshots_by_pack[pack_indices] = self._capture_snapshot(ctx)
        finally:
            # actor 恢复前先清本阶段临时 GPU contexts；快照已经 detach/clone 到 CPU。
            # 循环变量也会持有最后一个 context，不能只清列表。
            seq_ctx_list.clear()
            loss_ctx_list.clear()
            seq = ctx = None

        # 仅正常完成时恢复；物化/teacher/快照保存失败直接向外传播，不在 finally 中强行上卡。
        self.onload_model()
        self.onload_optimizer()
        return timings

    def _capture_snapshot(self, ctx: BaseRLLossContext) -> ActorSnapshot:
        """唯一冻结规则；只返回数据，不隐式读写 worker 缓存。"""
        kw = ctx.loss_kwargs
        values = {}
        for name in ActorSnapshot.__dataclass_fields__:
            if name in ("teacher_logprobs", "target_token_ids") and self._train_teacher_manager is None:
                values[name] = None         # rollout 静态目标由 full source 重建，不重复缓存
                continue
            tensor = getattr(kw, name, None)
            values[name] = None if tensor is None else tensor.detach().cpu().clone()
        return ActorSnapshot(**values)

    def _fit_policy_then_sft(
        self,
        batch: TrainBatch,
        snapshots_by_pack: dict[tuple[int, ...], ActorSnapshot],
    ) -> dict:
        """[S4] 先完成本轮 policy updates，再按既有频率执行一次 SFT tail。"""
        # 这里只展开训练日志；fit 中既有的 entropy / IS / teacher timings 汇总保持。
        worker_log_item = {"train_metrics": [], "sft_train_metrics": {}}
        for pack_indices in batch.plan.steps:
            # 每个 step 同步逐 pack 物化，只聚合该 step 的 ModelItems。
            cpu_packs = iter_cpu_packs(batch, pack_indices, self._resolve_refs)
            items = self._build_step_input(batch, cpu_packs, snapshots_by_pack)
            # 上一步已恢复冻结 snapshot，并完成 main/MTP 各自的一次 build_batches。
            # 所有 step 只读同一批快照；这里不重新计算 old/ref/teacher，也不再次校准 loss。
            # 本地 padding-only rank 仍参与；沿用 actor 原有的全局零监督约定。
            train_log_item = self._train_update(items)
            worker_log_item["train_metrics"].append(train_log_item)
            del items                       # 不将上一 step 的设备 contexts 带到下一次物化

        # rollout 计数按完成一轮 policy 训练增加一次，不按 microbatch / optimizer step 增加。
        # attempted/completed updates 与 scheduler 已由 RoleWorker._train_update 管理。
        self._rollout_step += 1
        if self._sft_dataloader is not None and self._rollout_step % self._rollout_steps_per_sft == 0:
            # 沿用 _fit_sft 的 dataloader/epoch/token 计数及独立 effective loss config。
            # 不传 TrainBatch 或 RL snapshot；SFT 按 S3 契约独立 build / build_batches。
            train_step_info = self._fit_sft()
            worker_log_item["sft_train_metrics"] = {
                **train_step_info["logs_info"],
                **train_step_info["extra_info"].get(),
                "efficient_attn_ratio": train_step_info["efficient_attn_ratio"],
            }
        return worker_log_item

    def _build_step_input(self, batch, full_packs, snapshots_by_pack):
        # full_packs 来自当前 optimizer step 的 iter_cpu_packs；只聚合这一 step 的 ModelItems。
        items = []
        for pack in full_packs:
            data = pack.full_data
            seq = data["seq_ctx"].copy().to(self.device)
            if self.sp_mesh.size() > 1:
                seq = seq.split(self.sp_mesh)
            main = self.loss_cfg.build(data, sp_mesh=self.sp_mesh, device=self.device)
            # MTP 始终读取 full source，与 main/IS 的构建顺序无关。
            mtp = [cfg.build(data, sp_mesh=self.sp_mesh, device=self.device) for cfg in self.mtp_cfgs]
            main.bind_actor_snapshot(snapshots_by_pack[pack.indices])
            items.append(ModelItem(seq_ctx=seq, loss_ctx={"lm": main, "mtp": mtp}))
        self.loss_cfg.loss_ctx_cls.build_batches([x.loss_ctx["lm"] for x in items])
        for depth, cfg in enumerate(self.mtp_cfgs):
            cfg.loss_ctx_cls.build_batches([x.loss_ctx["mtp"][depth] for x in items])
        return items


class CriticWorker(RoleWorker):
    role_name = "critic"

    def initialize(self):
        # [S4] S3 已拥有 actor→critic 例外；role 不再 strict=False。
        self.engine.from_hf(self.config.load_from, strict=True)

    def collect_values(self, batch: TrainBatch, *, return_values: bool):
        """[S4] SP-local forward → SP gather → 完整样本缓存。"""
        indices = [pack for step in batch.plan.steps for pack in step]
        rows = [None] * len(batch.plan.global_indices)
        self.engine._maybe_precompute_float8_dynamic_scale_for_fsdp()
        for pack in iter_cpu_packs(batch, indices, self._resolve_refs):
            full_seq = pack.full_data["seq_ctx"]
            pack_length = full_seq.input_ids.shape[1]
            seq = full_seq.copy().to(self.device)
            if self.sp_mesh.size() > 1:
                seq = seq.split(self.sp_mesh)
            with torch.no_grad():
                # 复用现有 critic raw value forward；不走 actor LogProbContext。
                local = self.forward_only(seq).float().squeeze(-1)
            full = sp_gather(local, self.sp_mesh, dim=1) if self.sp_mesh.size() > 1 else local
            full = full[:, :pack_length] # 先去 SP padding，再由 batch 去 pack padding
            for sample_id, value in batch.unpack_rows(pack, full):
                rows[sample_id] = value
            # 空 pack 同样 forward/gather，unpack 返回 []。
        batch.set_old_values(rows)       # 每个 replica 都有完整值，不仅代表 rank
        return TokenBatchProto.concat([row.flatten() for row in rows]) if return_values else None

    def compute_segment_targets(self, batch, estimator, *, contribute: bool):
        """[S4] 全 rank 参与；只允许一个 DP replica 贡献 full-row moments。"""
        trajectory = batch.trajectory_data()
        adv, ret = estimator.compute_segment_gae(**trajectory)
        moments = torch.zeros(3, dtype=torch.float64, device=self.device)
        if contribute:
            valid = adv[trajectory["mask"]]
            moments[:] = torch.tensor(
                [valid.double().sum(), valid.double().square().sum(), valid.numel()], device=self.device
            )
        if estimator.normalize_actor_advantage:
            if dist.is_initialized():
                dist.all_reduce(moments, op=dist.ReduceOp.SUM)  # 空 DP 也先参与
            adv = normalize_with_moments([adv], [trajectory["mask"]], moments.cpu())[0]
        adv_rows, ret_rows = EXISTING_SPLIT_TRAJECTORY_ROWS(adv, ret, trajectory["cu_seq_lens"])
        batch.set_ppo_targets(TokenBatchProto.concat(adv_rows), TokenBatchProto.concat(ret_rows))
        # 空代表 count=0，min/max 不贡献；非代表不向 Controller 重复贡献统计。
        return TokenStats.from_masked(adv, trajectory["mask"]) if contribute else TokenStats.empty()

    def fit(self, batch: TrainBatch):
        """[S4] PPO 要求 returns+old_values 完整；通用 critic MSE 不受此约束。"""
        # 此检查在完整 batch 的入口完成；不能 pack 内缺值时补0或补forward。
        batch.require_targets("critic")
        for packs in self._updates(batch):
            global_count = EXISTING_GLOBAL_SUPERVISED_COUNT(batch, packs)
            if global_count == 0:            # 所有 rank 得到同一判断；只本地空不能跳过
                continue
            items = []
            for pack in iter_cpu_packs(batch, packs, self._resolve_refs):
                data = pack.full_data
                seq = data["seq_ctx"].copy().to(self.device)
                if self.sp_mesh.size() > 1:
                    seq = seq.split(self.sp_mesh)
                ctx = self.loss_cfg.build(data, sp_mesh=self.sp_mesh, device=self.device)
                items.append(ModelItem(seq_ctx=seq, loss_ctx={"lm": ctx}))
            self.loss_cfg.loss_ctx_cls.build_batches([x.loss_ctx["lm"] for x in items])
            self._train_update(items)        # RoleWorker 的共享 optimizer Implementation

    def _updates(self, batch):
        """[S4] 每个 pass 都摊平重排后切分；per_pass=1 是该 pass 整批一次 update。"""
        packs = [pack for step in batch.plan.steps for pack in step]
        for pass_index in range(self.config.num_passes):
            # self.seed 由 TrainingWorker 构造时显式传入，不反查 host.config。
            order = pass_order(len(packs), batch.attrs.rollout_idx, pass_index, self.seed)
            for group in partition_indices(order, self.config.optimizer_steps_per_pass):
                yield [packs[i] for i in group]


# ============================================================================
# F. [S2 基础 + S4 扩展] controller.py：整段 target preparation，显式业务顺序
# ============================================================================


@dataclass(frozen=True)
class TokenStats:
    """[S4] 有效 action-token 口径；merge 忽略 count=0 的 extrema。"""
    sum: float
    count: int
    min: float | None
    max: float | None

    @classmethod
    def empty(cls):
        return cls(0.0, 0, None, None)

    @classmethod
    def from_masked(cls, values, mask):
        kept = values[mask]
        return cls(float(kept.sum()), kept.numel(), float(kept.min()), float(kept.max())) if kept.numel() else cls.empty()

    @classmethod
    def merge(cls, stats):
        valid = [s for s in stats if s.count]
        return cls(
            sum(s.sum for s in valid), sum(s.count for s in valid),
            min((s.min for s in valid), default=None), max((s.max for s in valid), default=None),
        )


class TrainingController:
    def __init__(self, workers, estimator, has_critic, topology):
        # [S4] 只在构造时按真实能力分派；不保存一份 _gae_reward_scope 影子值。
        if isinstance(estimator, TokenAdvantageEstimator) != has_critic:
            raise ValueError("Token estimator and critic must be configured together")
        self.workers, self.estimator, self.has_critic = workers, estimator, has_critic
        self.topology = topology             # 沿用既有 rank/DP replica 分组信息

    @contextmanager
    def _loaded_batch(self, prepared, plan):
        """[S2] payload refs / worker 本地 batch 生命周期；S4 critic 也在此 scope 内。

        不直接 free pixels/routed refs；成功返回意味着所有 worker 本地 batch 已关闭，
        RLTrainer 才能继续执行 I 节的远端资源释放。
        """
        handles, refs_by_dp = [], []
        try:
            # nested ref 避免 Ray 自动解引用；每 DP put 一次，所有 replicas 共享。
            for rank in plan.ranks:
                refs_by_dp.append([ray.put([prepared.states[i] for i in rank.global_indices])])
            for worker_index, worker in enumerate(self.workers):
                dp = self.topology.dp_rank(worker_index)
                handles.append(worker.load_train_batch.remote(refs_by_dp[dp], plan.ranks[dp], prepared.attrs))
            ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
            yield
        finally:
            # handles 只包含 load，不能拿它证明 yield 中全部业务 RPC 已结束。
            # 每个 worker 的 release 在该 worker 已提交的串行 RPC 后执行，
            # 物化/refs 解析都在该 RPC 内同步执行，没有额外后台预取任务需要等待。
            original_error = sys.exception()
            try:
                ray.get([w.release_train_batch.remote() for w in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)
            except Exception as cleanup_error:
                self.logger.warning("Batch cleanup failed: %s", cleanup_error)
                if original_error is None:
                    raise                   # 成功业务后的清理失败也交给 trainer abort
            finally:
                refs_by_dp.clear()          # 失败只丢普通 Ray 引用，不强删潜在在用数据
            # 原业务异常继续向外传播，RLTrainer 不执行成功路径的远端 refs 释放。
            # 失败退出若需显式 free，必须先停止仍在 collective/ray.get 内的消费者，不能直接套此成功流程。
            # 不在此恢复 actor 驻留；无 retry 的失败仍终止整轮，不新增 GPU barrier。

    def _prepare_ppo_targets(self, prepared, plan) -> TokenStats:
        """[S4] 后置条件：所有 worker 的 batch targets 完整；只返回统计。"""
        estimator = self.estimator
        session = estimator.reward_scope == "session"
        value_results = ray.get([
            w.critic_collect_values.remote(
                return_values=session and self.topology.is_dp_representative(i)
            ) for i, w in enumerate(self.workers)
        ])
        if not session:
            stats = ray.get([
                w.critic_compute_segment_targets.remote(
                    estimator, contribute=self.topology.is_dp_representative(i)
                ) for i, w in enumerate(self.workers)
            ])
            return TokenStats.merge(stats)

        rows_by_dp = [value_results[i] for i in self.topology.dp_representatives]
        values = plan.restore_rows(rows_by_dp)
        advantages, returns = session_token_targets(estimator, prepared.states, values)
        # 每 DP 仅构造/put 一次完整目标对；安装顺序与原样本 global indices 相同。
        targets_by_dp = []
        for rank in plan.ranks:
            targets_by_dp.append(ray.put((
                TokenBatchProto.concat([advantages[i] for i in rank.global_indices]),
                TokenBatchProto.concat([returns[i] for i in rank.global_indices]),
            )))
        ray.get([
            worker.set_ppo_targets.remote(targets_by_dp[self.topology.dp_rank(i)])
            for i, worker in enumerate(self.workers)
        ])
        return TokenStats.merge([
            TokenStats.from_masked(a, torch.tensor(state.labels[1:]).ne(-100))
            for state, a in zip(prepared.states, advantages)
        ])

    def fit(self, rollout_groups, rollout_idx):
        prepared = self._prepare_rollout_items(rollout_groups, rollout_idx)  # [S1]
        plan = self.packer.plan([s.num_tokens for s in prepared.states])  # [S2] 入口已验证训练长度。
        with self._loaded_batch(prepared, plan):
            critic_logs = None
            if self.has_critic:              # [S4] 保留真实业务分支，不造 NullCritic
                ray.get([w.set_residency.remote("critic", "critic") for w in self.workers])
                advantage_stats = self._prepare_ppo_targets(prepared, plan)
                critic_logs = ray.get([w.critic_fit.remote() for w in self.workers])
            else:
                # [S1/S2] 既有 group/session reward 映射 + scalar advantage 分发。
                # 保留 GRPO/OPD 原口径，不用 TokenStats 假装 scalar 也是 token 指标。
                advantage_stats = self._prepare_group_targets(prepared, plan)
            ray.get([w.set_residency.remote("actor", "actor") for w in self.workers])
            actor_logs = ray.get([w.actor_fit.remote() for w in self.workers])
        return self._build_trainer_log_info(prepared, plan, actor_logs, critic_logs, advantage_stats)


# ============================================================================
# G. [S4] worker.py：固定 actor/critic，具名 RPC；不增加 RoleResidency owner 表
# ============================================================================


class TrainingWorker(SingleAcceleratorWorker):
    """[S4] Ray 进程宿主；组合两个本地角色，与 RoleWorker 没有继承关系。

    初始化 PG/mesh、构造角色的既有流程保留；向 role 传必要配置与 mesh/device/rank/seed/logger，
    不传 host=self。角色独立构建自己的 engine，actor 先卸载再构造 critic 的内存顺序保持。
    训练与驻留 RPC 串行执行；角色正在 fit 时不能并发切换到另一个角色。
    """
    actor: ActorWorker
    critic: CriticWorker | None
    batch: TrainBatch | None                 # [S2] 唯一共享批次；role 只通过参数借用

    def load_train_batch(self, nested_ref, rank_plan, attrs):
        """[S2] 唯一 batch 安装入口；PPO 不新增第二份样本缓存。"""
        states = ray.get(nested_ref[0], timeout=TRAIN_RAY_GET_TIMEOUT)
        self.batch = TrainBatch(states, rank_plan, attrs, self.teacher_target_adapter)
        # [S2] attrs 携带全局 batch presence，空 DP 也能构造一致 padding/collective。
        # [S4] actor/critic 都只接收 self.batch；mesh/device/shared seed 从构造参数传入。

    def set_residency(self, model_role, optimizer_role):
        """[S4] 跨角色驻留决策；角色列表只有 actor + 可选 critic。

        验证角色配置在任何搬运之前完成；实际 placement 只由现有 engine 记录。
        先全部卸载 losers 再加载 winners，避免切换瞬间两个角色同时占目标资源。
        """
        roles = {r.role_name: r for r in (self.actor, self.critic) if r is not None}
        if model_role not in (*roles, None) or optimizer_role not in (*roles, None):
            raise ValueError("Requested role is not configured")
        for name, role in roles.items():
            if name != optimizer_role:
                role.offload_optimizer()
            if name != model_role:
                role.offload_model()
        if model_role is not None:
            roles[model_role].onload_model()
        if optimizer_role is not None:
            roles[optimizer_role].onload_optimizer()
        # 不写 _current_role / owner slots；中途异常沿用终止策略，不宣称迁移成功。

    def actor_fit(self):
        # 实际 model + optimizer placement 均需满足，不能只查 model owner。
        self.actor.require_train_residency()
        return self.actor.fit(self.batch)

    def critic_fit(self):
        self.critic.require_train_residency()
        return self.critic.fit(self.batch)

    def critic_collect_values(self, return_values):
        self.critic.require_model_residency()
        return self.critic.collect_values(self.batch, return_values=return_values)

    def critic_compute_segment_targets(self, estimator, contribute):
        return self.critic.compute_segment_targets(self.batch, estimator, contribute=contribute)

    def set_ppo_targets(self, payload):
        # Ray 顶层 ObjectRef 自动解引用；batch 内成对校验与安装。
        self.batch.set_ppo_targets(*payload)

    def release_train_batch(self):
        """[S2，S4 复用] 幂等本地释放；不 free 别的 replica 可能使用的 refs。"""
        if self.batch is not None:
            self.batch.close()
            self.batch = None

    # [S4] caller 迁移：
    # train actor          -> set_residency("actor", "actor")
    # weight update only   -> set_residency("actor", None)
    # yield to rollout     -> set_residency(None, None)
    # model-only offload   -> 从 engine 事实读取 optimizer role，set_residency(None, that_role)
    # optimizer-only offload -> 从 engine 事实读取 model role，set_residency(that_role, None)
    # teacher stage 在 actor.fit 内仅临时搬运 actor 自己的 engine，不回调此入口。
    # reference/teacher 自身模型沿用原管理 Module；host 不解释其内部阶段。


# ============================================================================
# H. [S4] role_worker.py / engine/train_engine.py：update 与完整续训
# ============================================================================


class RoleWorker(ABC):
    """[S4] 两个本地角色共用的单角色训练生命周期。

    各实例独立拥有 engine、scheduler、更新计数；不持有 host 或当前 TrainBatch。
    构造参数提供 mesh/device/rank/logger 等实际依赖；沿用既有 onload/offload 方法，
    它们只搬自己的 engine，不决定另一个角色的驻留。placement 以 engine 为准。
    fit 内的业务阶段、loss/refs 处理和更新分组由具体角色实现。
    """

    @abstractmethod
    def fit(self, batch: TrainBatch): ...

    # 展开既有搬运方法的核心；原有搬运耗时/显存日志省略，不新增宿主回调或驻留影子状态。
    def onload_model(self) -> bool:
        return self.engine.put_model_to_device(self.device)

    def onload_optimizer(self) -> bool:
        return self.engine.put_optimizer_to_device(self.device)

    def offload_model(self) -> bool:
        return self.engine.put_model_to_device("cpu")

    def offload_optimizer(self) -> bool:
        return self.engine.put_optimizer_to_device("cpu")

    def _train_update(self, items):
        self._global_train_step += 1
        train_step_info = self.engine.train_step(items)
        grad_norm = self.engine.clip_grad_norm()
        # [S4 engine 小扩展] 返回真实是否应用更新，删除 schedule.py 镜像 skip 条件。
        # 保留 engine 原 skip/zero_grad 语义；审计所有调用方再改返回契约。
        applied = self.engine.step_optimizer(grad_norm=grad_norm)
        if applied and self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # 保留既有 _train_optimizer_steps 的指标解析、loss_cfg.finalize_metrics、
        # grad_norm/显存记录和日志输出；这里的 EXISTING_* 表示省略的原代码，不新增转发 helper。
        return EXISTING_ROLE_STEP_LOG(self, train_step_info, grad_norm)

    def save(self, path: Path, *, save_optimizer=True):
        """[S4] path 已由 host 选择 actor/critic 子目录；只在 batch 完成点保存。"""
        self.engine.save_dcp(path, save_optimizer=save_optimizer)
        if self.rank == 0:
            torch.save({
                "global_train_step": self._global_train_step,
                "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                # LambdaLR.state_dict 不保存闭包参数；必须一起保存调度的定义。
                "scheduler_config": {"lr_cfg": self.config.lr_cfg.model_dump(),
                                     "scheduler_steps": self.config.scheduler_steps},
            }, path / "role_state.pt")
        # RNG 逐 rank：沿用 engine/trainer 的唯一 owner；缺失时补在角色状态中，不能双写。
        # Actor 专有 SFT dataloader 状态仍由 ActorWorker 保存。

    def resume(self, path: Path, cfg):
        # Interface 要求在新构造的 role 上 resume；初始 LR 才确实来自本次配置。
        initial_lrs = self.lr_scheduler.get_last_lr() if self.lr_scheduler is not None else None
        self.engine.load_dcp(path, load_states=cfg.load_optimizer_states, load_args=cfg.load_optimizer_args)
        state_file = path / "role_state.pt"
        state = torch.load(state_file, weights_only=True)
        if cfg.load_scheduler and self.lr_scheduler is not None:
            expected = {"lr_cfg": self.config.lr_cfg.model_dump(), "scheduler_steps": self.config.scheduler_steps}
            if state.get("scheduler_config") != expected:
                raise ValueError("Scheduler configuration changed; explicitly choose load_scheduler=False")
            self.lr_scheduler.load_state_dict(state["scheduler"])
        self._global_train_step = state["global_train_step"]
        if self.lr_scheduler is not None:
            lrs = self.lr_scheduler.get_last_lr() if cfg.load_scheduler else initial_lrs
            for group, lr in zip(self.engine.optimizer.param_groups, lrs):
                group["lr"] = lr            # optimizer args 加载不能覆盖所选 scheduler 语义
        # host 只选择 actor/critic 子目录；trainer 恢复 rollout/replay/RNG 后才发布完成。


# ============================================================================
# I. [S2，S4 复用] train/rl_trainer.py：已消费批次的远端 refs 释放
# ============================================================================


class RLTrainer:
    """仅展开现有 RLTrainer 的集成点；共卡与分离部署共用此释放路径。

    [S2] worker/role 不再显式 free routed refs，删除 WorkerConfig 中的
    free_rollout_routed_experts_in_worker 及 _init_train_worker_config 的条件赋值。
    原 _agent_loop_manager_uses_trace_store 只供该赋值使用，可随之删除；
    requires_rollout_proxy 的实际 rollout 配置用途保持。
    """

    def _train_one_batch(self, train_batch, train_step):
        ...                                  # 保留已有数据检查、轨迹保存和资源切换
        worker_logs, data_info = self.train_controller.fit(train_batch, rollout_idx=train_step)
        # Controller.fit 正常返回：actor/critic RPC 全部成功，所有 worker 的
        # release_train_batch 已确认；本批次不会再有 pack 读取远端 refs。
        # 若 fit/本地 batch release 失败，不进入这里；不能在外层 finally 无条件强删。
        self._release_train_batch_resources(train_batch)
        return {"data_info": data_info, "workers_log_item": worker_logs}

    def _release_train_batch_resources(self, train_batch):
        """[S2] 先按真实 session 释放 trace，再清剩余 routed/pixels refs。

        不根据 manager 的全局布尔值或 session_id 非空来猜测 trace 所有权。
        本次已消费的 session 全部 segments 必须都已结束；沿用现有按 group 消费语义。
        不支持把同一活跃 session 拆到未来 batch 后仍在此释放它。
        """
        from xtuner.v1.rl.utils import free_object_refs

        # 复用现有同步方法：只访问已存在的 store，返回实际存在且已释放的 session IDs。
        # 同一 session 的多个 segment 只请求一次；普通任务的 session_id 未必在 store 中。
        # RPC 失败直接传播，不能把失败冒充“没有 trace”，然后提前 free routed refs。
        released_sessions = self._release_trace_sessions(_trace_session_ids(train_batch))
        refs = []
        for group in train_batch:
            for state in group:
                pixels = state.mm_info.get("pixel_values") if state.mm_info is not None else None
                trace_released = state.session_id is not None and str(state.session_id) in released_sessions
                routed = None if trace_released else state.routed_experts
                # trace-owned routed refs 已随 session 被释放，不能在这里再次显式 free。
                # 非 trace、混合任务、debug 恢复且 store 无 session 的 routed refs 走本分支。
                for value in (pixels, routed):
                    if isinstance(value, ray.ObjectRef):
                        refs.append(value)
                    elif isinstance(value, (list, tuple)):
                        refs.extend(value)
        free_object_refs(refs)               # 既有 helper 自带 ObjectRef 筛选和去重

        # 替代训练成功路径原 _release_trace_sessions_after_train_batch 调用，不能再调用一次旧 hook。
        # 共卡原 release_all 改为同样仅释放本批次 sessions；分离部署不触碰后台 producer 的其他 sessions。
        # Eval 及 rejected/expired group 沿用各自既有清理流程；不将已完成训练样本标记为 discarded。


# 完整性检查（应落入未来真实测试，非本文件可执行测试）：
# S1：state.set_training_fields → replay consume → state.apply_response_mask → state.reset_response/retry。
# S1 序列化：真实 Ray 往返保留方法/metadata；debug snapshot/restore 支持 frozen dataclass。
# S2：public plan + TrainBatch materialize/unpack；MM/teacher/routes/empty packs。
# S2 资源：真实 trainer 成功/失败路径；混合 trace/普通任务、共享 segments、其他活跃 session 不受影响。
# S3：public build → model/engine，SP parity / source immutable / strict HF transitions。
# S4：真实 Controller.fit，scope/DP/session；IS snapshot；多 pass；save/rebuild/resume。
# 所有 GPU 测试先 gpu_lock.sh；本伪代码本身只做 AST 语法与文档一致性检查。
