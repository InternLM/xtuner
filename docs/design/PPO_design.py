"""PPO / 多 role TrainingController 伪代码。约定与阶段说明见 PPO_design.md。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

# xtuner/v1/model/base.py
# TransformerConfig
# 修改目标：增加 head_type="critic"，critic 主干复用 Transformer，输出头走 ValueHead。
class TransformerConfig:
    ...
    head_type: Literal["lm_head", "critic"] = "lm_head"


# xtuner/v1/config/fsdp.py
# FSDPConfig
# 修改目标：fp32_lm_head 改名为 fp32_head，同时约束 LMHead 与 ValueHead。
class FSDPConfig:
    fp32_head: bool = False


import torch
import torch.nn as nn
import torch.nn.functional as F


# ValueHead（新增）
# 修改目标：critic 标量 value 头，接口对齐 LMHead（loss_ctx is None 出 values，否则走 CriticLossContext）。
# 文件：新增，对齐 xtuner/v1/module/lm_head/lm_head.py。
class ValueHead(nn.Linear):
    """out_features=1。loss_ctx is None → values.float()；否则 CriticLossContext。"""

    def __init__(
        self,
        hidden_size: int,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """作用：构造 out_features=1 的线性头。文件：新增，对齐 xtuner/v1/module/lm_head/lm_head.py。"""
        super().__init__(
            in_features=hidden_size,
            out_features=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        loss_ctx: "CriticLossContext | None" = None,
    ) -> tuple[Any | None, tuple[Any | None, dict]]:
        """作用：无 loss_ctx 出 values.float()；有则走 CriticLossContext（value clip）。对齐 LMHead.forward。"""
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            b = self.bias.to_local() if self.bias is not None else None
        else:
            w, b = self.weight, self.bias
        if loss_ctx is None:
            values = F.linear(hidden_states, w, b)
            return None, (values.float(), {})
        else:
            return loss_ctx.forward(hidden_states, w, b)

    def _fully_shard(self, mesh) -> None:
        """作用：FSDP shard value 头。对齐 LMHead._fully_shard。"""
        ...


# TrainSample
# 修改目标：训练侧唯一协议。produce 只写原始字段；token_rewards / advantages 由 critic 或 GRPO 步内写。
# 文件：xtuner/v1/rl/data_proto/train_sample.py（新增）。
@dataclass
class TrainSample:
    input_ids: Any
    labels: Any
    response_mask: Any
    rollout_logprobs: Any | None
    routed_experts: Any | None
    group_id: int | None
    session_id: int | None
    extra_fields: dict[str, Any]
    num_tokens: int
    token_rewards: Any | None = None
    terminal_mask: Any | None = None
    reward: Any | None = None
    advantages: Any | None = None
    teacher_logprobs: Any | None = None
    target_token_ids: Any | None = None
    teacher_indices: Any | None = None



def build_train_sample(state: RolloutState) -> TrainSample:
    """作用：RolloutState → TrainSample。不写 token_rewards / terminal_mask / advantages。
    文件：xtuner/v1/rl/agent_loop_manager/produce_utils.py（及新增 train_sample.py）。"""
    input_ids = to_tensor(state.input_ids)
    teacher_logprobs, target_token_ids = align_rollout_teacher_targets(state)  # 无则 (None, None)
    return TrainSample(
        input_ids=input_ids,
        labels=to_tensor(state.labels),
        response_mask=to_tensor(state.response_mask),
        rollout_logprobs=to_tensor(state.logprobs),
        routed_experts=state.routed_experts,
        group_id=state.group_id,
        session_id=state.session_id,
        extra_fields=state.extra_fields,
        num_tokens=int(numel(input_ids)),
        reward=state.reward,
        teacher_logprobs=teacher_logprobs,
        target_token_ids=target_token_ids,
        teacher_indices=getattr(state, "teacher_indices", None),
    )


def build_train_samples(states: list[RolloutState]) -> list[TrainSample]:
    """作用：一组 RolloutState 批量转 TrainSample。文件：produce_utils.py。"""
    return [build_train_sample(s) for s in states]


# ProduceBatchResult
# 修改目标：训练入口只带 train_samples；删除 ProduceBatchResult.rollout_states 与 _prepare_train_data。
# 文件：xtuner/v1/rl/agent_loop_manager/produce_utils.py。
@dataclass
class ProduceBatchResult:
    train_samples: list[list[TrainSample]]
    ...


def build_produce_batch_result(..., batch_by_task, ...) -> ProduceBatchResult:
    """作用：produce 终点。groups → train_samples，不再携带 rollout_states。
    文件：xtuner/v1/rl/agent_loop_manager/produce_utils.py。"""
    groups = ...
    train_samples = [build_train_samples(group) for group in groups]
    return ProduceBatchResult(train_samples=train_samples, ...)


# FrozenWorkerResult
# 修改目标：Frozen forward 的返回协议。logprobs 必有；topk 另带 token_ids；driver 不解包 tensor。
# 文件：xtuner/v1/rl/trainer/worker.py。
@dataclass
class FrozenWorkerResult:
    logprobs: Any
    token_ids: Any | None = None
    logprobs_list: list[Tensor] | None = None
    token_ids_list: list[Tensor] | None = None


# PackMeta
# 修改目标：把 packing 方案从 Controller 内部提出；fit 建一次，多 role / 多算法共用同一 scatter。
# 文件：xtuner/v1/rl/trainer/controller.py（从 _get_pack_infos 抽出）。
@dataclass
class PackMeta:
    packs: list[dict]
    pack_max_length: int


def build_pack_meta(
    lengths: list[int],
    pack_max_length: int,
    *,
    random=None,
) -> PackMeta:
    """作用：按 num_tokens 贪心组 pack。从现有 TrainingController._get_pack_infos 抽出。
    文件：xtuner/v1/rl/trainer/controller.py。"""
    inds = list(range(len(lengths)))
    if random is not None:
        random.shuffle(inds)

    packs = []
    item_buffer, length_buffer, longest = [], [], 0
    for i in inds:
        if lengths[i] + sum(length_buffer) <= pack_max_length:
            item_buffer.append(i)
            length_buffer.append(lengths[i])
            longest = max(longest, lengths[i])
        else:
            if item_buffer:
                packs.append({"indices": item_buffer, "longest": int(longest)})
            item_buffer, length_buffer, longest = [i], [lengths[i]], lengths[i]
    if item_buffer:
        packs.append({"indices": item_buffer, "longest": int(longest)})
    return PackMeta(packs=packs, pack_max_length=pack_max_length)


def scatter_samples_and_meta(
    samples: list[TrainSample],
    pack_meta: PackMeta,
    workers: list,
) -> list[tuple[Any, PackMeta]]:
    """作用：按 DP 把 samples + 同一 PackMeta 切到各 rank。多 role 必须共用 fit 入口那份 meta。
    文件：xtuner/v1/rl/trainer/controller.py（TrainingWorkerGroup）。"""
    ...


def build_session_meta(
    sample_groups: list[list[TrainSample]],
) -> dict[str, Any]:
    """作用：按 session_id 建 segment→session。GRPO 用 session 级 adv；PPO 不拼 GAE 轨迹。
    文件：xtuner/v1/rl/trainer/controller.py。"""
    ...


def session_score(session_meta: dict[str, Any], sample: TrainSample) -> float:
    """作用：reward_scope=session 时取该 session representative 的 score。
    文件：xtuner/v1/rl/advantage/gae.py（或 critic 侧 helper）。"""
    ...


def get_pack_train_data(samples: list[TrainSample], pack_meta: PackMeta, **fields) -> list[dict]:
    """作用：按 PackMeta.packs 把 sample 字段物化成 packed batch。Critic/Frozen 不做 R3。
    文件：xtuner/v1/rl/trainer/worker.py。"""
    packed_batches = []
    for pack in pack_meta.packs:
        packed_batches.append({...})
    return packed_batches


# BaseWorkerConfig
# 修改目标：抽出 actor/critic/frozen 公共字段。pack_max_length 不在此，只挂 ActorWorkerConfig。
# 文件：xtuner/v1/rl/trainer/worker.py（从现有 WorkerConfig 拆出）。
class BaseWorkerConfig(BaseModel):
    model_cfg: Any  # TransformerConfig | BaseComposeConfig
    fsdp_cfg: Any
    load_from: str
    sp_size: int = 1
    log_dir: str | None = None
    seed: int | None = None

    def group_name(self) -> str:
        """作用：Controller 注册用的 group 名。子类必须实现。"""
        raise NotImplementedError


# ActorWorkerConfig
# 修改目标：现有 WorkerConfig 收口为 actor。独占 pack_max_length；ref_load_from 仅过渡，PPO KL 走 Frozen(reference)。
# 文件：xtuner/v1/rl/trainer/worker.py（现有 WorkerConfig）。
class ActorWorkerConfig(BaseWorkerConfig):
    pack_max_length: int
    optim_cfg: Any
    loss_cfg: Any
    lr_cfg: Any
    optimizer_steps: int = 1
    update_weight_bucket_size_in_gb: float = 0.5
    ref_load_from: str | None = None
    ref_model_fsdp_cfg: Any | None = None

    def group_name(self) -> str:
        """作用：固定返回 \"actor\"。"""
        return "actor"


# CriticWorkerConfig
# 修改目标：新增 critic 配置。无 pack_max_length，不 bind_rollout / weight_update。
# 文件：xtuner/v1/rl/trainer/worker.py。
class CriticWorkerConfig(BaseWorkerConfig):
    optim_cfg: Any
    loss_cfg: Any
    lr_cfg: Any
    optimizer_steps: int = 1

    def group_name(self) -> str:
        """作用：固定返回 \"critic\"。"""
        return "critic"


# FrozenWorkerConfig
# 修改目标：新增冻结模型配置。name 即 group 名（reference / teacher / teacher_*）；无 optim。
# 文件：xtuner/v1/rl/trainer/worker.py。
class FrozenWorkerConfig(BaseWorkerConfig):
    name: str = "reference"

    def group_name(self) -> str:
        """作用：用 self.name 作为 group 名，支持多个 Frozen。"""
        return self.name

    def is_teacher(self) -> bool:
        """作用：判断本 Frozen 是否 OPD teacher（name 为 teacher 或 teacher_*）。"""
        return self.name == "teacher" or self.name.startswith("teacher_")


# TrainConfig
# 修改目标：替换单 WorkerConfig。actor 必选；可选 critic 与 frozen 列表；advantage 注入 Controller。
# 文件：xtuner/v1/rl/trainer/worker.py（现有 train_worker_cfg）。
class TrainConfig(BaseModel):
    actor: ActorWorkerConfig
    critic: CriticWorkerConfig | None = None
    frozen: list[FrozenWorkerConfig] = Field(default_factory=list)
    advantage_estimator_config: Any

    def build(self, shared_pg) -> "TrainingController":
        """作用：在 train PG 上 spawn ActorWorker，attach critic/frozen，注册 groups。
        文件：xtuner/v1/rl/trainer/controller.py（或 TrainConfig.build）。"""
        return build_controller(self, shared_pg)


# GAEEstimator（新增）
# 修改目标：逐步 GAE。compute 必须按原 sample/segment，禁止作用在 pack concat 上。
# 文件：xtuner/v1/rl/advantage/gae.py（对齐 xtuner/v1/rl/advantage/grpo.py）。
class GAEEstimator:
    def __init__(self, gamma: float = 1.0, lam: float = 0.95):
        """作用：保存 γ、λ。"""
        self.gamma = gamma
        self.lam = lam

    def compute(
        self,
        token_rewards,
        old_values,
        response_mask,
        terminal_mask,
        bootstrap_value,
    ) -> tuple[Any, Any]:
        """作用：单条 sample 上算 advantages / returns。禁止对 pack concat 调用。"""
        ...


# GAEAdvantageConfig（新增）
# 修改目标：PPO 分发开关。reward_scope=segment|session 只改末 token 标量 r，不拼接多段轨迹。
# 文件：xtuner/v1/rl/advantage/gae.py。
class GAEAdvantageConfig:
    gamma: float = 1.0
    lam: float = 0.95
    reward_scope: Literal["segment", "session"] = "segment"

    def build(self) -> GAEEstimator:
        """作用：构造 GAEEstimator。reward_scope 留在 config 上给 critic.fit。"""
        return GAEEstimator(self.gamma, self.lam)


# CriticLossConfig（新增）
# 修改目标：critic value-clip 配置。不继承 BaseRLLossConfig，不挂 policy_loss_cfg。
# 文件：xtuner/v1/rl/loss/（新增，对齐 xtuner/v1/rl/loss/grpo_loss.py）。
class CriticLossConfig:
    cliprange_value: float = 0.2
    ignore_idx: int = -100

    @property
    def loss_ctx_cls(self) -> type["CriticLossContext"]:
        """作用：告诉 engine 用 CriticLossContext，对齐 LMHead / GRPO 的 loss_ctx_cls。"""
        return CriticLossContext

    @property
    def _loss_kwargs_cls(self) -> type["CriticLossKwargs"]:
        """作用：value loss kwargs 类型。"""
        return CriticLossKwargs


# CriticLossKwargs（新增）
# 修改目标：value loss 所需 returns / old_values / loss_weight，由 critic.fit 填入。
# 文件：xtuner/v1/rl/loss/。
class CriticLossKwargs:
    returns: Any
    old_values: Any
    loss_weight: Any


# CriticLossContext（新增）
# 修改目标：对齐 LMHead.loss_ctx。在 ValueHead.forward 内做 clipped value MSE。
# 文件：xtuner/v1/rl/loss/。
class CriticLossContext:
    def __init__(self, loss_cfg: CriticLossConfig, loss_kwargs: CriticLossKwargs):
        """作用：绑定 cliprange 与 returns/old_values/loss_weight。"""
        self.loss_cfg = loss_cfg
        self.loss_kwargs = loss_kwargs

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None = None,
    ) -> tuple[Any, tuple[Any | None, dict]]:
        """作用：clipped value MSE。对齐 LMHeadLossContext.forward。文件：xtuner/v1/rl/loss/。"""
        vpred = F.linear(hidden_states, head_weight, head_bias).float().squeeze(-1)
        old_v = self.loss_kwargs.old_values
        returns = self.loss_kwargs.returns
        w = self.loss_kwargs.loss_weight
        eps = self.loss_cfg.cliprange_value
        v_clip = old_v + (vpred - old_v).clamp(-eps, eps)
        per_token = 0.5 * torch.maximum((vpred - returns) ** 2, (v_clip - returns) ** 2)
        loss = (per_token * w).sum()
        return loss, (None, {})


from xtuner.v1.rl.utils import SingleAcceleratorWorker


# CriticWorker（新增，普通类）
# 修改目标：同进程 attach。fit 内 fill_gae → 按 sample GAE → value clip；不做 R3、不同步 rollout。
# 文件：xtuner/v1/rl/trainer/worker.py。
class CriticWorker:
    def __init__(self, cfg: CriticWorkerConfig):
        """作用：构建 critic engine（ValueHead）。不占用独立 Ray actor。"""
        self.cfg = cfg
        self.engine = build_critic_engine(cfg)

    def fill_gae_token_fields(
        self,
        samples: list[TrainSample],
        session_meta: dict[str, Any],
        reward_scope: Literal["segment", "session"],
    ) -> None:
        """作用：把稀疏 sequence reward 写成逐步 token_rewards / terminal_mask。
        segment 用 sample.reward；session 用 session_score。不拼多段轨迹。"""
        for s in samples:
            score = (
                session_score(session_meta, s)
                if reward_scope == "session"
                else float(s.reward["score"])
            )
            s.token_rewards = build_token_rewards(s, score)
            s.terminal_mask = build_terminal_mask(s)

    def forward_only(self, samples, pack_meta: PackMeta, **kw) -> tuple[Any, Any]:
        """作用：packed value forward，返回 old_values 与 bootstrap_value。不做 R3。"""
        packed = get_pack_train_data(samples, pack_meta, **kw)
        ...

    def fit(
        self,
        samples,
        pack_meta: PackMeta,
        gae: GAEEstimator,
        session_meta: dict[str, Any],
        reward_scope: Literal["segment", "session"],
    ) -> Any:
        """作用：fill_gae → packed V → 按 sample 切分 GAE → value clip 反传。返回 per-rank advantages。"""
        self.fill_gae_token_fields(samples, session_meta, reward_scope)
        old_v_packed, boot_packed = self.forward_only(samples, pack_meta)
        advantages_chunks, returns_chunks, old_v_chunks = [], [], []
        for s, v_s, boot_s in zip(
            samples,
            split_by_sample(old_v_packed, pack_meta, samples),
            split_bootstrap(boot_packed, pack_meta, samples),
        ):
            adv_s, ret_s = gae.compute(
                s.token_rewards, v_s, s.response_mask, s.terminal_mask, boot_s,
            )
            advantages_chunks.append(adv_s)
            returns_chunks.append(ret_s)
            old_v_chunks.append(v_s)
        returns = cat_chunks_to_pack(returns_chunks, pack_meta)
        old_v = cat_chunks_to_pack(old_v_chunks, pack_meta)
        packed = get_pack_train_data(
            samples, pack_meta, returns=returns, old_values=old_v,
        )
        ...
        return advantages_chunks

    def onload(self):
        """作用：本进程 critic onload。由 switch_role_modules 触发。"""
        ...

    def offload(self):
        """作用：本进程 critic offload。"""
        ...

    def save(self, path: str, **kw):
        """作用：保存 critic 权重。不走 rollout sync。"""
        ...

    def resume(self, load_cfg):
        """作用：恢复 critic checkpoint。"""
        ...


# FrozenWorker（新增，普通类）
# 修改目标：同进程 attach 的 reference / teacher。forward_only(logprob|sampled|topk)；Teacher 不做 R3。
# 文件：xtuner/v1/rl/trainer/worker.py。
class FrozenWorker:
    def __init__(self, cfg: FrozenWorkerConfig):
        """作用：加载冻结 LM（requires_grad=False）。name 决定 reference 还是 teacher。"""
        self.cfg = cfg
        self.name = cfg.name
        self.model = build_frozen_model(cfg.model_cfg, cfg.load_from, cfg.fsdp_cfg)

    def forward_only(
        self,
        samples: list[TrainSample],
        pack_meta: PackMeta,
        *,
        mode: Literal["logprob", "sampled", "topk"] = "logprob",
        top_k: int | None = None,
    ) -> FrozenWorkerResult:
        """作用：logprob=PPO KL；sampled/topk=OPD teacher 目标。不做 R3。"""
        packed = get_pack_train_data(samples, pack_meta)
        self.model.to_device(DEVICE)
        try:
            if mode == "topk":
                assert top_k is not None
                token_ids, logprobs = compute_topk_teacher_targets(self.model, packed, top_k=top_k)
                return FrozenWorkerResult(logprobs=logprobs, token_ids=token_ids)
            if mode == "sampled":
                logprobs = compute_sampled_teacher_logprobs(self.model, packed)
                return FrozenWorkerResult(logprobs=logprobs, token_ids=None)
            logprobs = compute_token_logprobs(self.model, packed)
            return FrozenWorkerResult(logprobs=logprobs, token_ids=None)
        finally:
            self.model.to_device("cpu")

    def onload(self):
        """作用：冻结模型 onload。"""
        ...

    def offload(self):
        """作用：冻结模型 offload。"""
        ...


# ActorWorker
# 修改目标：唯一 Ray remote。宿主 attach critic/frozen；PG、R3、weight_update 只走本类；复用 GRPOLossConfig。
# 文件：xtuner/v1/rl/trainer/worker.py（现有 TrainingWorker 收口）。
class ActorWorker(SingleAcceleratorWorker):
    def __init__(
        self,
        worker_cfg: ActorWorkerConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        """作用：Ray 进程入口；建 actor engine 与 _attached 字典。基类：xtuner/v1/rl/utils/ray_accelerator_worker.py。"""
        super().__init__(worker_cfg, rank, master_addr, master_port, world_size, accelerator)
        self.cfg = worker_cfg
        self.engine = build_actor_engine(worker_cfg)
        self._attached: dict[str, Any] = {}

    def ready(self):
        """作用：Ray spawn 完成握手。对齐现有 TrainingWorker.ready。"""
        ...

    def attach(self, name: str, builder, cfg) -> None:
        """作用：在本进程构造 CriticWorker / FrozenWorker，写入 _attached。"""
        self._attached[name] = builder(cfg)

    def call_attached(self, name: str, method: str, *args, **kwargs):
        """作用：TrainingWorkerGroup(attached=True) 按 name 转发到同进程 worker。"""
        return getattr(self._attached[name], method)(*args, **kwargs)

    def forward_only(self, samples, pack_meta: PackMeta, **kw) -> Any:
        """作用：冻 π_old logprobs（PPO ratio 锚点，≠ rollout_logprobs）。可走 R3。"""
        packed = get_pack_train_data(samples, pack_meta, **kw)
        ...

    def fit(self, samples, pack_meta: PackMeta, *, rollout_idx: int, **frozen_stats) -> dict:
        """作用：clipped PG（复用 GRPOLossConfig）。PPO 吃 advantages/old_lp/ref_lp；OPD 吃 teacher_*。"""
        packed = get_pack_train_data(samples, pack_meta, **frozen_stats)
        ...

    def weight_update(self, **kw):
        """作用：仅 actor 权重推到 rollout。文件：现有 worker.weight_update。"""
        ...

    def bind_rollout_weight_update(self, *, targets, rollout_config):
        """作用：绑定 NCCL 同步目标。仅 actor。"""
        ...

    def switch_role_modules(self, role: str) -> None:
        """作用：步内分时 onload/offload。role=actor|critic|reference|teacher[_*]|none。≠ Trainer 的 rollout↔train。"""
        ...

    def onload(self, target="all"):
        """作用：train 侧 onload（Trainer 在 fit 外调用）。"""
        ...

    def offload(self, target="all"):
        """作用：train 侧 offload。"""
        ...

    def save(self, path: str, **kw):
        """作用：保存 actor；可再转发 attached critic.save。"""
        ...

    def resume(self, load_cfg):
        """作用：恢复 actor（及 attached）checkpoint。"""
        ...


# TrainingWorkerGroup
# 修改目标：按 name 路由 actor 直调 vs call_attached。各 group 共用 ActorWorker handles 与同一 PackMeta scatter。
# 文件：xtuner/v1/rl/trainer/controller.py。
class TrainingWorkerGroup:

    def __init__(
        self,
        name: str,
        workers: list,
        *,
        attached: bool = False,
        can_sync_rollout: bool = False,
    ):
        """作用：包装同一批 ActorWorker Ray handles。attached=True 时走 call_attached。"""
        self.name = name
        self.workers = workers
        self.attached = attached
        self.can_sync_rollout = can_sync_rollout

    def _scatter_call(self, method: str, samples, pack_meta: PackMeta, *args, **kwargs):
        """作用：scatter 后按 rank 调 worker.method。等长 list kwargs 按 rank 切开。不解包业务 tensor。"""
        shards = scatter_samples_and_meta(samples, pack_meta, self.workers)
        n = len(self.workers)

        def _rank_kwargs(i: int) -> dict:
            """作用：把与 workers 等长的 kwargs 切到第 i 卡。"""
            out = {}
            for k, v in kwargs.items():
                if isinstance(v, (list, tuple)) and len(v) == n:
                    out[k] = v[i]
                else:
                    out[k] = v
            return out

        handles = []
        for i, (w, (s, m)) in enumerate(zip(self.workers, shards)):
            kw = _rank_kwargs(i)
            if self.attached:
                handles.append(w.call_attached.remote(self.name, method, s, m, *args, **kw))
            else:
                handles.append(getattr(w, method).remote(s, m, *args, **kw))
        return handles

    def forward_only(self, samples, pack_meta: PackMeta, **kwargs):
        """作用：转发 forward_only，返回 per-rank ObjectRef。driver 不解包。"""
        return self._scatter_call("forward_only", samples, pack_meta, **kwargs)

    def fit(self, samples, pack_meta: PackMeta, **kwargs):
        """作用：actor 则 ray.get 聚合 logs；critic 返回 advantages 句柄不解包。"""
        handles = self._scatter_call("fit", samples, pack_meta, **kwargs)
        if self.name == "critic":
            return handles
        return ray_get(handles)

    def broadcast_host(self, method: str, *args, **kwargs):
        """作用：进程级广播（switch_role_modules / onload / weight_update），只打 ActorWorker。"""
        handles = [getattr(w, method).remote(*args, **kwargs) for w in self.workers]
        return ray_get(handles)


# TrainingController
# 修改目标：对外仍 fit；对内按配置转发 GRPO/PPO/OPD。本步唯一 PackMeta；步内 switch_role_modules，不建 PG。
# 文件：xtuner/v1/rl/trainer/controller.py。
class TrainingController:
    def __init__(self, advantage_estimator_config: Any, *, pack_max_length: int):
        """作用：持有 groups、advantage、actor.pack_max_length。"""
        self._groups: dict[str, TrainingWorkerGroup] = {}
        self._adv_cfg = advantage_estimator_config
        self._advantage_estimator = advantage_estimator_config.build()
        self.pack_max_length = pack_max_length  # = actor.pack_max_length

    def register(self, group: TrainingWorkerGroup) -> None:
        """作用：按 group.name 注册，禁止重名。"""
        if group.name in self._groups:
            raise ValueError(f"duplicate group: {group.name}")
        self._groups[group.name] = group

    def group(self, name: str) -> TrainingWorkerGroup:
        """作用：按名取 group。"""
        return self._groups[name]

    def switch_role_modules(self, role: str):
        """作用：步内分时。≠ RLTrainer 的 rollout↔train（后者在 fit 外）。"""
        self._groups["actor"].broadcast_host("switch_role_modules", role)

    def onload(self, target="all"):
        """作用：Trainer 在 fit 前 train onload。文件：现有 TrainingController.onload。"""
        self._groups["actor"].broadcast_host("onload", target)

    def offload(self, target="all"):
        """作用：Trainer 在 fit 后 train offload。"""
        self._groups["actor"].broadcast_host("offload", target)

    def weight_update(self, **kw):
        """作用：只推 actor → rollout。critic 不参与。"""
        g = self._groups["actor"]
        assert g.can_sync_rollout
        g.broadcast_host("weight_update", **kw)

    def _teacher_names(self) -> list[str]:
        """作用：已注册的 teacher / teacher_* group 名。推理引擎 Teacher 时为空。"""
        return [
            n for n in self._groups
            if n == "teacher" or n.startswith("teacher_")
        ]

    def fit(
        self,
        sample_groups: list[list[TrainSample]],
        *,
        rollout_idx: int,
    ):
        """作用：Trainer 唯一训练入口。建 session_meta + 本步唯一 PackMeta，再转发 PPO/OPD/GRPO。
        入参已是可训 group：produce 侧过滤对齐现有 _prepare_train_data（fit 不再二次过滤）。
        文件：xtuner/v1/rl/trainer/controller.py；Trainer 侧删 xtuner/v1/train/rl_trainer.py 的 _prepare_train_data。"""
        session_meta = build_session_meta(sample_groups)
        samples = [s for g in sample_groups for s in g]
        lengths = [s.num_tokens for s in samples]
        pack_meta = build_pack_meta(lengths, self.pack_max_length)

        if isinstance(self._adv_cfg, GAEAdvantageConfig):
            return self.train_ppo_batch(
                sample_groups,
                pack_meta=pack_meta,
                session_meta=session_meta,
                rollout_idx=rollout_idx,
            )
        rollout_teacher_ready = bool(samples) and samples[0].teacher_logprobs is not None
        if rollout_teacher_ready or self._teacher_names():
            return self.train_opd_batch(
                sample_groups,
                pack_meta=pack_meta,
                session_meta=session_meta,
                rollout_idx=rollout_idx,
            )
        return self.train_grpo_batch(
            sample_groups,
            pack_meta=pack_meta,
            session_meta=session_meta,
            rollout_idx=rollout_idx,
        )

    def train_grpo_batch(
        self,
        sample_groups: list[list[TrainSample]],
        *,
        pack_meta: PackMeta,
        session_meta: dict[str, Any],
        rollout_idx: int,
    ):
        """作用：session 级 GRPO adv 再广播回 segment，然后 actor.fit。对齐现有 GRPOEstimator.compute。
        文件：controller.fit 内联；advantage 现实现于 xtuner/v1/rl/advantage/grpo.py。"""
        self.switch_role_modules("actor")
        for session in session_meta["sessions"]:
            reps = session["representatives"]  # 每 session 一个 reward 代表
            rewards = torch.tensor(
                [float(s.reward["score"]) for s in reps], dtype=torch.float32
            )
            advantages = self._advantage_estimator.compute(rewards, reps)  # (S,)
            for seg, adv in zip(session["segments"], broadcast_to_segments(advantages, session)):
                seg.advantages = adv  # 标量；pack 时广播到 label!=-100
        samples = [s for g in sample_groups for s in g]
        return self._groups["actor"].fit(samples, pack_meta, rollout_idx=rollout_idx)

    def train_ppo_batch(
        self,
        sample_groups: list[list[TrainSample]],
        *,
        pack_meta: PackMeta,
        session_meta: dict[str, Any],
        rollout_idx: int,
    ):
        """作用：critic.fit(GAE) → 可选 ref logprob → actor 冻 π_old → actor.fit(clipped PG)。"""
        assert "critic" in self._groups
        samples = [s for g in sample_groups for s in g]
        self.switch_role_modules("critic")
        adv_refs = self._groups["critic"].fit(
            samples,
            pack_meta,
            gae=self._advantage_estimator,
            session_meta=session_meta,
            reward_scope=self._adv_cfg.reward_scope,
        )
        ref_refs = None
        if "reference" in self._groups:
            self.switch_role_modules("reference")
            ref_refs = self._groups["reference"].forward_only(samples, pack_meta, mode="logprob")
        self.switch_role_modules("actor")
        old_lp_refs = self._groups["actor"].forward_only(samples, pack_meta)
        return self._groups["actor"].fit(
            samples,
            pack_meta,
            advantages=adv_refs,
            old_logprobs=old_lp_refs,
            ref_logprobs=ref_refs,
            rollout_idx=rollout_idx,
        )

    def train_opd_batch(
        self,
        sample_groups: list[list[TrainSample]],
        *,
        pack_meta: PackMeta,
        session_meta: dict[str, Any],
        rollout_idx: int,
        teacher_mode: Literal["sampled", "topk"] = "sampled",
        top_k: int | None = None,
    ):
        """作用：推理 Teacher 则用 sample.teacher_*；否则遍历 teacher[_*] forward_only，再 actor 蒸馏。"""
        samples = [s for g in sample_groups for s in g]
        teacher_refs: dict[str, Any] | None = None
        if samples[0].teacher_logprobs is None:
            teacher_names = self._teacher_names()
            assert teacher_names
            teacher_refs = {}
            for name in teacher_names:
                self.switch_role_modules(name)
                teacher_refs[name] = self._groups[name].forward_only(
                    samples, pack_meta, mode=teacher_mode, top_k=top_k,
                )
        self.switch_role_modules("actor")
        return self._groups["actor"].fit(
            samples, pack_meta, teacher_targets=teacher_refs, rollout_idx=rollout_idx,
        )


def spawn_actor_on_pg(cfg: ActorWorkerConfig, train_pg):
    """作用：唯一 Ray spawn。Colocate 绑 shared_pg；Disagg 绑 train_pg。
    文件：对齐 xtuner/v1/rl/utils/ray_accelerator_worker.py 的 AutoAcceleratorWorkers。"""
    ActorRemote = ray.remote(
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
            }
        }
    )(ActorWorker)
    workers, _ = AutoAcceleratorWorkers.from_placement_group(ActorRemote, cfg, train_pg)
    ray.wait([w.ready.remote() for w in workers])
    return workers


def build_controller(train_cfg: TrainConfig, train_pg) -> TrainingController:
    """作用：spawn actor → attach critic/frozen → 注册 groups。TrainConfig.build 的实现。"""
    actors = spawn_actor_on_pg(train_cfg.actor, train_pg)

    if train_cfg.critic is not None:
        ray.get([w.attach.remote("critic", CriticWorker, train_cfg.critic) for w in actors])
    for frozen_model in train_cfg.frozen:
        ray.get([
            w.attach.remote(frozen_model.group_name(), FrozenWorker, frozen_model)
            for w in actors
        ])

    ctl = TrainingController(
        advantage_estimator_config=train_cfg.advantage_estimator_config,
        pack_max_length=train_cfg.actor.pack_max_length,
    )
    ctl.register(TrainingWorkerGroup("actor", actors, can_sync_rollout=True))
    if train_cfg.critic is not None:
        ctl.register(TrainingWorkerGroup("critic", actors, attached=True))
    for frozen_model in train_cfg.frozen:
        ctl.register(TrainingWorkerGroup(frozen_model.group_name(), actors, attached=True))
    return ctl


def example_grpo_trainer_config():
    """作用：GRPO 配置草图。文件风格对齐 examples/v1/config/rl_*.py。"""
    from copy import deepcopy

    actor_model_cfg = deepcopy(base_model_cfg)
    actor_model_cfg.head_type = "lm_head"

    actor = ActorWorkerConfig(
        model_cfg=actor_model_cfg,
        load_from=model_path,
        optim_cfg=AdamWConfig(lr=1e-6, foreach=False),
        loss_cfg=GRPOLossConfig(
            policy_loss_cfg=dict(
                cliprange_high=0.28,
                cliprange_low=0.2,
                loss_type="vanilla",
            ),
            use_kl_loss=False,
        ),
        lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
        fsdp_cfg=FSDPConfig(torch_compile=False, cpu_offload=False),
        pack_max_length=pack_max_length,
        optimizer_steps=1,
        sp_size=1,
    )

    train_worker_cfg = TrainConfig(
        actor=actor,
        critic=None,
        frozen=[],
        advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
    )
    trainer = RLColocateTrainerConfig(
        resources=resources,
        train_worker_cfg=train_worker_cfg,
        rollout_config=rollout_config,
        tokenizer_path=model_path,
        load_from=model_path,
        train_batch_size=train_batch_size,
        work_dir=work_dir,
        seed=123,
    )
    return trainer


def example_ppo_trainer_config():
    """作用：PPO 配置草图（actor+critic+reference，reward_scope=segment）。"""
    from copy import deepcopy

    actor_model_cfg = deepcopy(base_model_cfg)
    actor_model_cfg.head_type = "lm_head"

    actor = ActorWorkerConfig(
        model_cfg=actor_model_cfg,
        load_from=model_path,
        optim_cfg=AdamWConfig(lr=1e-6, foreach=False),
        loss_cfg=GRPOLossConfig(
            policy_loss_cfg=dict(
                cliprange_high=0.2,
                cliprange_low=0.2,
                loss_type="vanilla",
            ),
            use_kl_loss=True,
            kl_loss_coef=0.001,
            kl_loss_type="low_var_kl",
        ),
        lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
        fsdp_cfg=FSDPConfig(torch_compile=False, cpu_offload=False),
        pack_max_length=pack_max_length,
        optimizer_steps=1,
    )

    critic_model_cfg = deepcopy(base_model_cfg)
    critic_model_cfg.head_type = "critic"
    # critic 不支持 MTP
    if hasattr(critic_model_cfg, "mtp_config"):
        critic_model_cfg.mtp_config = None

    critic = CriticWorkerConfig(
        model_cfg=critic_model_cfg,
        load_from=model_path,
        optim_cfg=AdamWConfig(lr=1e-5, foreach=False),
        loss_cfg=CriticLossConfig(cliprange_value=0.2),
        lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
        fsdp_cfg=FSDPConfig(torch_compile=False, cpu_offload=False, recompute_ratio=0),
        optimizer_steps=1,
    )

    ref_model_cfg = deepcopy(base_model_cfg)
    ref_model_cfg.head_type = "lm_head"

    reference = FrozenWorkerConfig(
        name="reference",
        model_cfg=ref_model_cfg,
        load_from=model_path,
        fsdp_cfg=FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False),
    )

    train_worker_cfg = TrainConfig(
        actor=actor,
        critic=critic,
        frozen=[reference],
        advantage_estimator_config=GAEAdvantageConfig(
            gamma=1.0, lam=0.95, reward_scope="segment",
        ),
    )
    trainer = RLColocateTrainerConfig(
        resources=resources,
        train_worker_cfg=train_worker_cfg,
        rollout_config=rollout_config,
        tokenizer_path=model_path,
        load_from=model_path,
        train_batch_size=train_batch_size,
        work_dir=work_dir,
        seed=123,
    )
    return trainer


def example_opd_trainer_config():
    """作用：训练引擎多 teacher（MOPD）配置草图。推理 Teacher 则 frozen=[] 且 produce 预填 teacher_*。"""
    from copy import deepcopy

    actor = ActorWorkerConfig(
        model_cfg=deepcopy(base_model_cfg),
        load_from=model_path,
        optim_cfg=AdamWConfig(lr=1e-6, foreach=False),
        loss_cfg=...,
        lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
        fsdp_cfg=FSDPConfig(torch_compile=False, cpu_offload=False),
        pack_max_length=pack_max_length,
    )
    teachers = [
        FrozenWorkerConfig(
            name="teacher_0",
            model_cfg=deepcopy(teacher_model_cfg),
            load_from=teacher_path_0,
            fsdp_cfg=FSDPConfig(recompute_ratio=0, requires_grad=False),
        ),
        FrozenWorkerConfig(
            name="teacher_1",
            model_cfg=deepcopy(teacher_model_cfg),
            load_from=teacher_path_1,
            fsdp_cfg=FSDPConfig(recompute_ratio=0, requires_grad=False),
        ),
    ]
    train_worker_cfg = TrainConfig(
        actor=actor,
        critic=None,
        frozen=teachers,
        advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
    )
    return RLColocateTrainerConfig(
        resources=resources,
        train_worker_cfg=train_worker_cfg,
        rollout_config=rollout_config,
        load_from=model_path,
        train_batch_size=train_batch_size,
    )
