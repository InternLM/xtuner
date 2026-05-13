import asyncio
import importlib
import inspect
import os
import re
import traceback
from typing import Callable, List, Self, Tuple

import ray
from lagent.serving.sandbox.providers.gateway import GatewayProvider

from xtuner.v1.data_proto.rl_data import (
    RLDataFlowItem,
    RLJudgerResponseItem,
    RolloutState,
    update_dataflow_item,
)
from xtuner.v1.ray.environment.lagent.schema import AgentMessage
from xtuner.v1.ray.environment.rl_task.schemas import SandboxSpec
from xtuner.v1.ray.environment.trace import emit_fate, init_writer, span
from xtuner.v1.utils import get_logger

from .base_env import BaseEnvironment


def check_dead_actors():
    # 获取所有 Actor 的列表
    from ray.util.state import list_actors

    all_actors = list_actors()

    dead_actors = []
    for actor_info in all_actors:
        # 状态通常是 "ALIVE", "DEAD", "RECONSTRUCTING" 等
        if actor_info["state"] == "DEAD":
            dead_actors.append(actor_info)

    return dead_actors


DEFAULT_GATEWAY = "http://env-gateway.ailab.ailab.ai"
# DEFAULT_LAGENT_SRC = "/mnt/shared-storage-user/llmit/user/wangziyi/projs/lagent"
DEFAULT_LAGENT_SRC = "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent"


def _import_from_path(path: str):
    """Import an object from a dotted path like 'pkg.mod.attr'."""
    if not path or not isinstance(path, str):
        raise TypeError(f"pipeline must be a non-empty str, got {type(path)}")
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid import path: {path!r}. Expected 'module.attr'.")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _resolve_pipeline(pipeline_spec, sandbox_spec: dict | None = None):
    """Resolve pipeline spec into a Runner-like object with run_single()."""
    obj = _import_from_path(pipeline_spec)
    # Most configs point to a factory function (e.g., claw_pipeline()).
    if sandbox_spec is not None:
        return obj(sandbox=SandboxSpec(**sandbox_spec))
    else:
        return obj()


def _sample_task_id(sample: RLDataFlowItem) -> str | None:
    """Best-effort task_id extraction for trace emission — safe even when the
    sample ended up failing before run_single populated its result."""
    try:
        extra = getattr(sample.data, "extra_info", None) or {}
        td = extra.get("task_data") if isinstance(extra, dict) else None
        return (
            str(td.id)
            if td is not None and hasattr(td, "id")
            else extra.get("task_id")
            if isinstance(extra, dict)
            else None
        )
    except Exception:
        return None


_RE_RETURN_CODE = re.compile(r"return_code=(-?\d+)")


def _classify_mark_failed_reason(reason: str | None) -> str:
    """Bucketize a ``_mark_failed`` reason string into a finer ``failed_stage``
    label for fates.jsonl.  Falls back to ``mark_failed`` when no pattern
    matches — that should be rare and worth chasing.

    Args:
        reason (str | None): The ``failure_reason`` string produced by
            :func:`runner._mark_failed` (or its caller).

    Returns:
        str: Short kebab-style label used as the fate ``failed_stage``.
    """
    r = reason or ""
    # Legacy watchdog label — shouldn't fire under the detach+poll model
    # but keep the bucket so historical data lands somewhere meaningful.
    if "DaemonStuckError" in r:
        return "daemon_stuck"
    if "could not acquire" in r or "sandbox after" in r:
        return "acquire_failed"
    # Post-hook / pre-hook failures wrapped by ``_run_hook`` as
    # ``RuntimeError: <phase>-hook <HookType>('<name>') image=... failed: ...``
    # The most common rc22 symptom: entry rc=0 but ``/tmp/message.json``
    # 404s on download (sandbox died between entry and post-hook, or
    # lagent_entry exited without producing the file).
    if "post-hook" in r:
        if "404 Not Found" in r:
            return "posthook_download_404"
        if "Source file does not exist" in r:
            return "posthook_file_missing"
        return "posthook_failed"
    if "pre-hook" in r:
        return "prehook_failed"
    m = _RE_RETURN_CODE.search(r)
    if m:
        rc = int(m.group(1))
        if rc == -1:
            return "entry_pid_lost"
        if rc == -2:
            return "entry_daemon_gone"
        if rc == -3:
            return "entry_timeout"
        if rc == -9 or rc == 137:
            return "oom_killed"
        if rc > 0:
            return f"entry_rc_{rc}"
    if "TimeoutError" in r:
        return "timeout"
    if "OutOfMemoryError" in r or "OutOfMemory" in r:
        return "oom"
    return "mark_failed"


@ray.remote(  # type: ignore[call-overload]
    max_concurrency=int(os.environ.get("XTUNER_MAX_CONCURRENCY", 2000)),
    scheduling_strategy="SPREAD",
)
class InstallAgentEnvironment(BaseEnvironment):
    def __init__(
        self,
        environment: str,
        rollout_controller,
        preprocess_func: Callable[[Self, RLDataFlowItem], Tuple[AgentMessage]] = lambda _, item: (
            AgentMessage(role="user", content=item.data.messages[0]["content"]),  # type: ignore[index]
        ),
        postprocess_func: Callable[[Self, List[RLDataFlowItem]], List[RLDataFlowItem]] = lambda _, items: items,
    ):
        super().__init__(environment, None, None, None, None)
        self.rollout_controller = rollout_controller
        self.provider = GatewayProvider(DEFAULT_GATEWAY)
        self.preprocess_func = preprocess_func
        self.postprocess_func = postprocess_func
        # Trace writer is per-actor-process; safe to call twice (later calls
        # are no-ops).  WORK_DIR unset → emission is silently disabled.
        try:
            actor_id = ray.get_runtime_context().get_actor_id()
        except Exception:
            actor_id = None
        init_writer(actor_id=actor_id)

    async def generate(  # type: ignore[override]
        self, group_data_items: List[RLDataFlowItem], sample_params=None, extra_params=None
    ) -> List[RLDataFlowItem]:
        sample_params = sample_params.model_dump() if sample_params else {}

        async def _inner_agent_call(item):
            if item.env.rollout.state == RolloutState.COMPLETED:
                get_logger().debug(f"Rollout already completed for item {item.uid.observation_id}, skip agent call.")
                return "Passed"

            # agent_inputs = self.preprocess_func(self, deepcopy(item))
            pipeline = item.data.extra_info.get("pipeline", None)
            sandbox_spec = item.data.extra_info.get("sandbox_spec", None)
            pipeline = _resolve_pipeline(pipeline, sandbox_spec)

            task_dir = item.data.extra_info.get("task_dir", None)
            data = item.data.extra_info.get("task_data", None)
            uid = {
                "root_id": item.uid.root_id,
                "action_id": item.uid.action_id,
                "observation_id": item.uid.observation_id,
            }
            get_logger().info(f"running task={task_dir} (dataset={data.id})")

            uid_obs = str(item.uid.observation_id)
            with span(uid_obs, "env_sample_total", task_id=_sample_task_id(item)):
                try:
                    return await pipeline.run_single(
                        task_dir,
                        data,
                        uid,
                        provider=self.provider,
                        lagent_src_dir=DEFAULT_LAGENT_SRC,
                        llm_model=os.environ.get("RL_LLM_MODEL"),
                        llm_base_url=os.environ.get("RL_LLM_BASE_URL"),
                        llm_api_key=os.environ.get("RL_LLM_API_KEY"),
                    )
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    get_logger().error(
                        f"[Agent Inference Error] {exc}. Dead actors: {check_dead_actors()}\n{traceback.format_exc()}"
                    )
                    # Stash the exception type + message so the classification
                    # loop can bucket the fate by ``runner_exc/<ExceptionType>``
                    # instead of the generic ``runner_exc/pipeline-exception``.
                    return ("Failed", type(exc).__name__, str(exc))

        # Every sample emits exactly one fate to ``$WORK_DIR/trace/fates.*.jsonl``
        # before ``generate`` returns.  Track which uids have already fired so
        # the outer ``except CancelledError`` block can emit ``CANCELLED`` only
        # for the still-pending ones.  The ``try`` must wrap ``asyncio.gather``
        # itself — DataFlow's step-end ``ray.cancel`` raises CancelledError
        # inside gather, and if the try started below gather we would silently
        # lose the cancelled samples (observed in rc19: 102 CancelledError
        # spans but 0 fates).
        emitted_uids: set[str] = set()
        try:
            results = await asyncio.gather(*[_inner_agent_call(sample) for sample in group_data_items])
            passed_data_items, completed_data_items, skipped_data_items = [], [], []
            # Every failure path marks the sample SKIPPED (state + fate) and
            # routes it into ``skipped_data_items``.  ``skipped_data_items`` is
            # **not** included in the return value — only ``passed`` +
            # ``completed_data_items_result`` go back to dataflow.  This lets a
            # partially-broken group proceed with whatever siblings survived
            # instead of getting the whole group vetoed by
            # ``determine_group_state`` on seeing a SKIPPED state in the
            # returned list.  The tradeoff: GRPO/RLOO baselines now see
            # variable-size groups (1..prompt_repeat_k) — the downstream RL
            # code must compute baselines from whatever samples are present.
            #
            # Each SKIPPED warning carries the sample's action_id (= group id
            # shared by all prompt_repeat_k members).  Ray driver-log dedup can
            # fold many uid-level lines into one summary; action_id lets us
            # reconstruct group-level SKIP counts from the survivors.
            for sample, result in zip(group_data_items, results):
                group_id = sample.uid.action_id
                uid_obs = str(sample.uid.observation_id)
                group_id_str = str(group_id)
                sample_task_id = _sample_task_id(sample)
                if isinstance(result, tuple) and result and result[0] == "Failed":
                    # _inner_agent_call's own except already logged the traceback.
                    # The tuple carries the exception type and message so the
                    # fate lands in ``runner_exc/<ExceptionType>`` — far more
                    # useful than the old generic ``runner_exc`` bucket.
                    _, exc_type, exc_msg = result
                    get_logger().warning(
                        f"[InstallAgentEnvironment] sample SKIPPED uid={sample.uid.observation_id} "
                        f"group_id={group_id} task_id=? reason=runner_exc/{exc_type}"
                    )
                    emit_fate(
                        uid=uid_obs,
                        task_id=sample_task_id,
                        group_id=group_id_str,
                        final="SKIPPED",
                        failed_stage=f"runner_exc/{exc_type}",
                        reason=f"{exc_type}: {exc_msg[:400]}",
                    )
                    emitted_uids.add(uid_obs)
                    sample.env.rollout.state = RolloutState.SKIPPED
                    skipped_data_items.append(sample)
                    continue
                if result == "Passed":
                    passed_data_items.append(sample)
                    continue
                finish_reason = result["env"]["rollout"]["finish_reason"]
                task_id = (result.get("data") or {}).get("extra_info", {}).get("task_id")
                extra = result["env"]["rollout"].get("extra_info") or {}
                if finish_reason == "failed":
                    # Structured fields from runner.py::_infer_metadata —
                    # the runner directly knows which stage failed and with
                    # what rc.  Regex fallback only kicks in if those fields
                    # are absent (e.g. older fates, unusual code paths).
                    failure_reason = extra.get("failure_reason", "unknown")
                    fine_stage = extra.get("failed_stage") or _classify_mark_failed_reason(failure_reason)
                    fate_fields = {
                        k: extra.get(k)
                        for k in (
                            "entry_rc",
                            "entry_stderr",
                            "exception_type",
                            "sandbox_image",
                            "sandbox_env_id",
                            "sandbox_url",
                        )
                        if extra.get(k) is not None
                    }
                    get_logger().warning(
                        f"[InstallAgentEnvironment] sample SKIPPED uid={sample.uid.observation_id} "
                        f"group_id={group_id} task_id={task_id} finish_reason=failed "
                        f"stage={fine_stage} entry_rc={extra.get('entry_rc')} "
                        f"sandbox_url={extra.get('sandbox_url')} reason={failure_reason}"
                    )
                    emit_fate(
                        uid=uid_obs,
                        task_id=task_id or sample_task_id,
                        group_id=group_id_str,
                        final="SKIPPED",
                        failed_stage=fine_stage,
                        reason=failure_reason,
                        **fate_fields,
                    )
                    emitted_uids.add(uid_obs)
                    sample.env.rollout.state = RolloutState.SKIPPED
                    skipped_data_items.append(sample)
                    continue
                if finish_reason == "abort":
                    get_logger().warning(
                        f"[InstallAgentEnvironment] sample SKIPPED uid={sample.uid.observation_id} "
                        f"group_id={group_id} task_id={task_id} finish_reason=abort"
                    )
                    emit_fate(
                        uid=uid_obs,
                        task_id=task_id or sample_task_id,
                        group_id=group_id_str,
                        final="SKIPPED",
                        failed_stage="aborted",
                        reason="abort",
                    )
                    emitted_uids.add(uid_obs)
                    sample.env.rollout.state = RolloutState.SKIPPED
                    skipped_data_items.append(sample)
                    continue
                # Defend against silent-pass / truncated trajectory: rc=0 but
                # last message in policy_agent.messages lacks the fields
                # postprocess will read.  Same heuristic as
                # DumpDaemonLogOnFailure so log signal + filter stay consistent.
                msg_dict = (result["env"]["agent"] or {}).get("message_dict") or {}
                messages = msg_dict.get("policy_agent.messages") or []
                last = messages[-1] if messages else {}
                required = ("raw_content", "raw_content_ids", "raw_content_logprobs")
                missing = [k for k in required if not last.get(k)]
                if missing:
                    get_logger().warning(
                        f"[InstallAgentEnvironment] sample SKIPPED uid={sample.uid.observation_id} "
                        f"group_id={group_id} task_id={task_id} reason=silent-pass missing={missing}"
                    )
                    emit_fate(
                        uid=uid_obs,
                        task_id=task_id or sample_task_id,
                        group_id=group_id_str,
                        final="SKIPPED",
                        failed_stage="silent_pass",
                        reason=f"missing={missing}",
                    )
                    emitted_uids.add(uid_obs)
                    sample.env.rollout.state = RolloutState.SKIPPED
                    skipped_data_items.append(sample)
                    continue
                sample.env.agent.extra_info["message_dict"] = msg_dict
                sample.env.agent.extra_info["daemon_log"] = result["env"]["agent"].get("daemon_log", "")
                sample.env.judger.extra_info.update(result["env"]["judger"])
                completed_data_items.append(sample)

            # Wrap user postprocess — any raise would kill the trainer. See
            # AgentEnvironment.generate for rationale.  On batch failure we fall
            # back to per-item invocation so only the broken sample is marked
            # SKIPPED instead of the whole group; batch-level postprocess usually
            # has a per-item loop inside, so a single malformed item should not
            # take down the rest.
            async def _invoke_postprocess(items):
                out = self.postprocess_func(self, items)  # type: ignore[arg-type]
                if inspect.iscoroutine(out):
                    out = await out
                return out

            try:
                completed_data_items_result = await _invoke_postprocess(completed_data_items)
            except Exception as exc:
                import traceback as _tb

                get_logger().warning(
                    f"[InstallAgentEnvironment] postprocess_func failed on batch of "
                    f"{len(completed_data_items)} samples, falling back to per-item: "
                    f"{exc}\n{_tb.format_exc()}"
                )
                completed_data_items_result = []
                for item in completed_data_items:
                    try:
                        out = await _invoke_postprocess([item])
                    except Exception as per_item_exc:
                        get_logger().warning(
                            f"[InstallAgentEnvironment] postprocess_func failed on "
                            f"uid={item.uid.observation_id}: "
                            f"{type(per_item_exc).__name__}: {per_item_exc}"
                        )
                        emit_fate(
                            uid=str(item.uid.observation_id),
                            task_id=_sample_task_id(item),
                            group_id=str(item.uid.action_id),
                            final="SKIPPED",
                            failed_stage="postprocess",
                            reason=f"{type(per_item_exc).__name__}: {per_item_exc}",
                        )
                        emitted_uids.add(str(item.uid.observation_id))
                        item.env.rollout.state = RolloutState.SKIPPED
                        # Route to skipped_data_items so the caller's return
                        # (``passed + completed_data_items_result``) keeps the
                        # failed sample out of the downstream group state —
                        # otherwise a stray SKIPPED state inside the result
                        # would trigger ``determine_group_state`` veto on an
                        # otherwise-salvageable partial group.
                        skipped_data_items.append(item)
                    else:
                        completed_data_items_result.extend(out)
            # Every surviving item in ``completed_data_items_result`` finished
            # both rollout and postprocess cleanly — emit COMPLETED fate.
            # Siblings that failed emitted their SKIPPED fate at the relevant
            # branch above; they're in ``skipped_data_items`` and NOT in the
            # returned list, so the group is no longer vetoed by their death.
            for item in completed_data_items_result:
                uid_obs = str(item.uid.observation_id)
                emit_fate(
                    uid=uid_obs,
                    task_id=_sample_task_id(item),
                    group_id=str(item.uid.action_id),
                    final="COMPLETED",
                )
                emitted_uids.add(uid_obs)
            return passed_data_items + completed_data_items_result
        except asyncio.CancelledError:
            # Step-end forced cancel (DataFlow ray.cancel) or upstream abort:
            # emit CANCELLED for any sample that hasn't already been tagged
            # so fates.jsonl never has a silent hole.
            for sample in group_data_items:
                uid_obs = str(sample.uid.observation_id)
                if uid_obs in emitted_uids:
                    continue
                emit_fate(
                    uid=uid_obs,
                    task_id=_sample_task_id(sample),
                    group_id=str(sample.uid.action_id),
                    final="CANCELLED",
                    failed_stage="cancel",
                    reason="step-end or external cancel",
                )
                emitted_uids.add(uid_obs)
            raise

    async def run(  # type: ignore[override]
        self, group_data_items: List[RLDataFlowItem], sample_params=None, extra_params=None
    ) -> List[RLDataFlowItem]:
        group_data_items = await self.generate(group_data_items, sample_params, extra_params)
        skip_judger = any(
            item.env.rollout.finish_reason == "abort" or item.env.rollout.finish_reason == "failed"
            for item in group_data_items
        )
        if self.judger_controller and not skip_judger:
            try:
                judger_responses: List[RLJudgerResponseItem] = await asyncio.wait_for(
                    self.judger_controller.run.remote(group_data_items), timeout=1200.0
                )
            except asyncio.TimeoutError:
                judger_responses = [RLJudgerResponseItem(extra_info={"state": "failed"}) for _ in group_data_items]
            group_data_items = update_dataflow_item(group_data_items, "env.judger", judger_responses)
        return group_data_items
