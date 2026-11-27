"""Multi-turn tool-calling agent loop.

The loop appends tool calls and tool results in the rollout token space. Tools are
injected through ``tool_executor_factory``, so the loop does not depend on a task
specific data schema or tool implementation.

Design:
- Each turn calls ``rollout_ctl.generate`` and keeps the rollout token space in
  placeholder form; the engine expands image placeholders, so tool images need no
  rollout-side change.
- Tool-result wrappers follow the model-family chat template: text results use
  ``tool_wrap``; image results append one ``image_markup`` placeholder after the
  text. All boundaries are special tokens.
- Training fields are assembled manually: ``train_prompt_ids`` (the expanded
  dataset prompt) plus per-turn gen_ids/wrapper_ids; labels cover only generated
  tokens, and logprobs are aligned per generated token with zeros elsewhere.
  Image tool results expand in-place to N image_token_id tokens
  (N = prod(grid) / merge_unit) and extend mm_info
  (pixel_values/grid/num_img_tokens plus a full-sequence mm_token_type_ids
  aligned with TrainBatch input_ids[:-1], i.e. (1, seq_len - 1)).
"""

from __future__ import annotations

import base64
import io
import json
import re
import traceback
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import ray

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop.agent_loop import AgentLoop, AgentLoopConfig
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.trace.rollout_api import trace_rollout_remote
from xtuner.v1.utils.logger import get_logger


logger = get_logger()


@dataclass
class ToolResult:
    """A tool call result.

    Args:
        text (str): Text returned to the policy.
        image_data (str | None): Optional image data URI returned to the policy.
    """

    text: str
    image_data: str | None = None


class ToolExecutor(Protocol):
    """Callable tool set used by :class:`MultiTurnToolAgentLoop`."""

    def call(self, name: str, arguments: dict[str, str]) -> ToolResult: ...

    def close(self) -> None: ...


ToolExecutorFactory = Callable[[RolloutState], ToolExecutor]

OBS_TOKEN = "<|observation|>"
TOOL_WRAP_TEMPLATE = "<|observation|><tool_response>{}</tool_response><|assistant|><think>"
IMAGE_MARKUP = "<|begin_of_image|><|image|><|end_of_image|>"
IMG_WRAP_PREFIX_TEMPLATE = "<|observation|><tool_response>{}<|begin_of_image|>"
IMG_WRAP_SUFFIX = "<|end_of_image|></tool_response><|assistant|><think>"
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*([A-Za-z_][\w]*)\s*((?:<arg_key>.*?</arg_key>\s*<arg_value>.*?</arg_value>\s*)*)</tool_call>", re.S
)
ARG_RE = re.compile(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>", re.S)
QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
FORCE_CALL_RE = re.compile(r"^([A-Za-z_]\w*)\(([^)]*)\)$")


@dataclass(frozen=True)
class _FormatProfile:
    """Model-family chat boundary conventions.

    ``tool_wrap`` / ``img_wrap_*`` are aligned with the model chat templates; ``image_markup`` must encode to exactly
    one image token.
    """

    name: str
    stop_token: str  # 引擎在工具轮可能一并产出的停止符（产出则剥掉，wrapper 会重补）
    tool_wrap: str  # 文本工具结果 wrapper 模板，{} 为结果文本
    img_wrap_prefix: str  # 图片工具结果 wrapper 前缀模板，{} 为结果文本，末尾为 begin 标记
    img_wrap_suffix: str
    image_markup: str  # rollout 空间单占位符标记（自检必须恰好编码出 1 个 image_token_id）
    has_mm_flags: bool  # GLM 需全序列 mm_token_type_ids；Qwen 训练侧从 input_ids 反推，无需标记


PROFILES: dict[str, _FormatProfile] = {
    "glm53": _FormatProfile(
        name="glm53",
        stop_token=OBS_TOKEN,
        tool_wrap=TOOL_WRAP_TEMPLATE,
        img_wrap_prefix=IMG_WRAP_PREFIX_TEMPLATE,
        img_wrap_suffix=IMG_WRAP_SUFFIX,
        image_markup=IMAGE_MARKUP,
        has_mm_flags=True,
    ),
    "qwen35_vl": _FormatProfile(
        name="qwen35_vl",
        stop_token="<|im_end|>",
        tool_wrap="<|im_end|>\n<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n<|im_start|>assistant\n",
        img_wrap_prefix="<|im_end|>\n<|im_start|>user\n<tool_response>\n{}\n<|vision_start|>",
        img_wrap_suffix="<|vision_end|>\n</tool_response><|im_end|>\n<|im_start|>assistant\n",
        image_markup="<|vision_start|><|image_pad|><|vision_end|>",
        has_mm_flags=False,
    ),
}


def _parse_forced_call(spec: str) -> tuple[str, dict[str, str]]:
    """解析强制注入条目，格式 "name(key=value,...)"，如 "get_page_image(page=1)"。"""
    match = FORCE_CALL_RE.match(spec.strip())
    if match is None:
        raise ValueError(f"invalid force_tool_calls entry: {spec!r}")
    args: dict[str, str] = {}
    for part in match.group(2).split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"invalid force_tool_calls arg {part!r} in {spec!r}")
        args[key.strip()] = value.strip()
    return match.group(1), args


_PROCESSOR_CACHE: dict[str, Any] = {}


def _load_processor(hf_checkpoint: str):
    if hf_checkpoint not in _PROCESSOR_CACHE:
        from transformers import AutoProcessor

        _PROCESSOR_CACHE[hf_checkpoint] = AutoProcessor.from_pretrained(hf_checkpoint, trust_remote_code=True)
    return _PROCESSOR_CACHE[hf_checkpoint]


def parse_tool_call(text: str) -> tuple[str, dict[str, str]] | None:
    """GLM 风格：从生成文本解析首个工具调用；没有则返回 None。"""
    match = TOOL_CALL_RE.search(text)
    if match is None:
        return None
    name = match.group(1)
    args = dict(ARG_RE.findall(match.group(2)))
    return name, args


def parse_qwen_tool_call(text: str) -> tuple[str, dict[str, str]] | None:
    """Parse a Qwen-style JSON tool call."""
    match = QWEN_TOOL_CALL_RE.search(text)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = obj.get("arguments") or {}
    if not isinstance(arguments, dict):
        return None
    return name, {str(k): str(v) for k, v in arguments.items()}


class MultiTurnToolAgentLoopConfig(AgentLoopConfig):
    """Configuration for :class:`MultiTurnToolAgentLoop`.

    Args:
        max_turns (int): Maximum number of generation turns.
        force_tool_calls (list[str]): Tool calls injected when the policy emits no
            parseable call, used to exercise the data path in smoke tests.
        model_family (str): Key of the tool-result wrapper profile.
        max_total_tokens (int | None): Hard cap on the assembled trajectory length.
        final_turn_hint (bool): Append a wrap-up hint on the last tool turn.
        strict_think_tool_parse (bool): Parse tool calls only after a closed think block.
    """

    max_turns: int = 6
    force_tool_calls: list[str] = []
    model_family: str = "glm53"
    max_total_tokens: int | None = None
    final_turn_hint: bool = False
    strict_think_tool_parse: bool = False

    @abstractmethod
    def build_tool_executor_factory(self) -> ToolExecutorFactory:
        """Build a per-sample tool executor factory."""

    def build_local(
        self,
        rollout_controller: RolloutController | None = None,
        judger: Judger | None = None,
        logger=None,
    ) -> MultiTurnToolAgentLoop:
        return MultiTurnToolAgentLoop(
            rollout_controller,
            self.sample_params,
            self.hf_checkpoint,
            judger=judger,
            logger=logger,
            enable_batch_judge=self.enable_batch_judge,
            max_turns=self.max_turns,
            force_tool_calls=list(self.force_tool_calls),
            model_family=self.model_family,
            max_total_tokens=self.max_total_tokens,
            final_turn_hint=self.final_turn_hint,
            strict_think_tool_parse=self.strict_think_tool_parse,
            tool_executor_factory=self.build_tool_executor_factory(),
        )


FINAL_TURN_WRAPUP_NOTE = (
    "\n\n[Note: Tool budget exhausted. Do not output any <tool_call>; "
    "provide your complete final answer directly based on all information gathered so far.]"
)
_MIN_TURN_TOKENS = 64


class MultiTurnToolAgentLoop(AgentLoop):
    """Run a multi-turn tool-calling rollout and assemble training fields.

    Args:
        rollout_ctl (RolloutController | None): Rollout controller.
        sample_params (SampleParams | None): Default sampling parameters.
        hf_checkpoint (str): Hugging Face checkpoint used for tokenizer and processor.
        judger (Judger | None): Optional judger.
        logger: Optional logger.
        enable_batch_judge (bool): Whether to batch judge completed samples.
        max_turns (int): Maximum number of generation turns.
        force_tool_calls (list[str] | None): Forced tool calls for contract tests.
        model_family (str): Key of the tool-result wrapper profile.
        max_total_tokens (int | None): Hard cap on the assembled trajectory length.
        final_turn_hint (bool): Append a wrap-up hint on the last tool turn.
        strict_think_tool_parse (bool): Parse tool calls only after a closed think block.
        tool_executor_factory (ToolExecutorFactory | None): Per-sample tool executor factory.
    """

    def __init__(
        self,
        rollout_ctl: RolloutController | None,
        sample_params,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger=None,
        enable_batch_judge: bool = True,
        max_turns: int = 6,
        force_tool_calls: list[str] | None = None,
        model_family: str = "glm53",
        max_total_tokens: int | None = None,
        final_turn_hint: bool = False,
        strict_think_tool_parse: bool = False,
        tool_executor_factory: ToolExecutorFactory | None = None,
    ):
        super().__init__(rollout_ctl, sample_params, hf_checkpoint, judger, logger, enable_batch_judge)
        if tool_executor_factory is None:
            raise ValueError("MultiTurnToolAgentLoop requires a tool_executor_factory.")
        self.tool_executor_factory = tool_executor_factory
        self.max_turns = max_turns
        self.max_total_tokens = max_total_tokens
        self.final_turn_hint = final_turn_hint
        self.strict_think_tool_parse = strict_think_tool_parse
        # smoke 兜底：policy（减层模型）输出乱码不会自发产生可解析的 <tool_call>，
        # 配置 force_tool_calls 时在解析失败的轮次按序强制注入，用于验证数据通路。
        # 语义说明：注入的调用不是 policy 的行为，只验证 wrapper/两空间记账/引擎展开/
        # 训练契约/judger 透传链路；extra_fields["agent_forced_tool_calls"] 记录注入次数。
        self._forced_specs = [_parse_forced_call(spec) for spec in (force_tool_calls or [])]
        if model_family not in PROFILES:
            raise ValueError(f"unknown model_family {model_family!r}, expected one of {sorted(PROFILES)}")
        self._fmt = PROFILES[model_family]
        self._parse = parse_qwen_tool_call if model_family == "qwen35_vl" else parse_tool_call
        self._stop_id = self.tokenizer.convert_tokens_to_ids(self._fmt.stop_token)
        self._think_start_id = self._single_token_id("<think>")
        self._think_end_id = self._single_token_id("</think>")
        self._tool_end_id = self._single_token_id("</tool_call>")
        # 不变量自检：wrapper 编码必须以停止符开头
        probe = self.tokenizer.encode(self._fmt.tool_wrap.format("x"), add_special_tokens=False)
        assert probe[0] == self._stop_id, f"tool wrapper encoding misaligned: {probe[:4]}"
        # 图片占位符自检：image_markup 必须恰好编码出 1 个 image_token_id
        processor = _load_processor(hf_checkpoint)
        self._processor = processor
        self._image_token_id = int(processor.image_token_id)
        markup_ids = self.tokenizer.encode(self._fmt.image_markup, add_special_tokens=False)
        assert markup_ids.count(self._image_token_id) == 1, f"image markup encoding misaligned: {markup_ids}"
        self._merge_unit = int(processor.image_processor.merge_size) ** 2

    def _single_token_id(self, token_str: str) -> int | None:
        ids = self.tokenizer.encode(token_str, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def _fail(self, state: RolloutState, msg: str) -> RolloutState:
        state.status = Status.FAILED
        state.finish_reason = "error"
        state.error_msg = msg
        logger.warning(f"[MultiTurnToolAgentLoop] sample rollout_id={state.rollout_id} failed: {msg}")
        return state

    def _process_tool_image(self, data_uri: str) -> tuple[np.ndarray, np.ndarray, int]:
        """Data URI -> (pixel_values, image_grid_thw, num_tokens)，与 dataset
        侧展开配方一致。"""
        from PIL import Image

        png_bytes = base64.b64decode(data_uri.split(",", 1)[1])
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        output = self._processor.image_processor(images=[image], return_tensors="pt")
        pixel_values = output["pixel_values"]
        if pixel_values.ndim == 3 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values[0]
        pixel_values = pixel_values.numpy()
        grid = output["image_grid_thw"][0].numpy()
        num_tokens = int(np.prod(grid)) // self._merge_unit
        return pixel_values, grid, num_tokens

    def _extend_mm_info(
        self,
        state: RolloutState,
        tail_flags: list[int],
        tail_pixels: list[np.ndarray],
        tail_grids: list[np.ndarray],
        tail_num_tokens: list[int],
    ) -> None:
        """Merge tool image media fields into ``state.mm_info``.

        With ``has_mm_flags`` (GLM) the mm_token_type_ids is extended to the full sequence; Qwen derives visual
        positions from input_ids and only appends pixel/grid/num_img_tokens.
        """
        base: dict[str, Any] = dict(state.mm_info or {})
        assert state.input_ids is not None
        assert state.prompt_ids is not None
        seq_len = len(state.input_ids)
        prompt_len = len(state.prompt_ids)
        if self._fmt.has_mm_flags:
            base_mm = base.get("mm_token_type_ids")
            if base_mm is not None:
                prompt_flags = np.asarray(base_mm).reshape(-1)
                assert prompt_flags.shape[0] == prompt_len, (
                    f"mm_token_type_ids prompt length mismatch: {prompt_flags.shape[0]} vs {prompt_len}"
                )
            else:
                prompt_flags = np.zeros(prompt_len, dtype=np.int64)
            full_flags = np.concatenate([prompt_flags, np.asarray(tail_flags, dtype=np.int64)])[None, :]
            assert full_flags.shape[1] == seq_len
            # TrainBatch 训练输入是 input_ids[:-1]（与 labels[1:]/logprobs[1:] 同语义），
            # 全序列契约期望 (1, seq_len - 1)；末位 token 必是生成文本（flag=0），截掉对齐。
            assert full_flags[0, -1] == 0, "last token must be generated text with flag 0"
            base["mm_token_type_ids"] = full_flags[:, :-1]
        base["num_img_tokens"] = list(base.get("num_img_tokens") or []) + tail_num_tokens
        if tail_pixels:
            new_pixels = np.concatenate(tail_pixels, axis=0)
            base_pixels = base.get("pixel_values")
            if base_pixels is not None and not isinstance(base_pixels, np.ndarray):
                # sampler.put_to_ray 把像素字段换成 RayObjectRef（训练侧契约，
                # resolve_mm_inputs 要求 ObjectRef），合并前先取回真实数组。
                base_pixels = ray.get(base_pixels)
            base["pixel_values"] = (
                new_pixels if base_pixels is None else np.concatenate([np.asarray(base_pixels), new_pixels], axis=0)
            )
            if ray.is_initialized():
                base["pixel_values"] = ray.put(base["pixel_values"])
            new_grids = np.stack(tail_grids, axis=0)
            base["image_grid_thw"] = (
                new_grids
                if base.get("image_grid_thw") is None
                else np.concatenate([np.asarray(base["image_grid_thw"]), new_grids], axis=0)
            )
        state.mm_info = base  # type: ignore[assignment]

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        state = rollout_state
        train_prompt = state.extra_fields.get("train_prompt_ids")
        if not train_prompt:
            return self._fail(
                state, "train_prompt_ids missing: MultiTurnToolAgentLoop requires multimodal prompt expansion"
            )
        if state.session_id is None:
            import uuid

            state.session_id = uuid.uuid4().int

        try:
            tools = self.tool_executor_factory(state)
        except Exception as exc:
            return self._fail(state, f"tool executor init failed: {exc}")

        assert state.prompt_ids is not None
        cumulative = list(state.prompt_ids)  # rollout 空间（占位符形式）
        appended: list[int] = []  # train 空间 initial prompt 之后的追加 token
        labels_tail: list[int] = []  # 与 appended 对齐的 labels
        logprobs_tail: list[float] = []  # 与 appended 对齐的 logprobs（非生成位为 0）
        tail_mm_flags: list[int] = []  # 与 appended 对齐的模态标记（0 文本/生成，1 图片）
        tail_pixels: list[np.ndarray] = []  # 工具图片 pixel_values，按出现顺序
        tail_grids: list[np.ndarray] = []  # 工具图片 image_grid_thw，按出现顺序
        tail_num_tokens: list[int] = []  # 工具图片展开 token 数，按出现顺序
        tool_images: list[dict[str, Any]] = []  # 工具图片 data URI + 页码，按出现顺序
        prompt_image_data = list(state.extra_fields.get("image_data") or [])
        forced_queue = list(self._forced_specs)
        n_forced = 0
        n_tool_calls = 0
        n_truncated_turns = 0
        last_finish_reason = "stop"
        final_text = ""

        try:
            for turn in range(1, self.max_turns + 1):
                turn_state = state.model_copy(deep=True)
                turn_state.tokens = list(cumulative)
                turn_state.response_ids = None
                turn_state.logprobs = None
                turn_state.routed_experts = None
                if self.final_turn_hint and turn == self.max_turns and self.max_turns >= 2:
                    turn_state.tools = None
                    turn_state.tool_choice = None
                if self.max_total_tokens is not None:
                    cur_train_len = len(train_prompt) + len(appended)
                    remaining_tokens = self.max_total_tokens - cur_train_len
                    if remaining_tokens < _MIN_TURN_TOKENS:
                        return self._fail(
                            state,
                            f"turn {turn}: trajectory length {cur_train_len} exhausted max_total_tokens {self.max_total_tokens}",
                        )
                    if remaining_tokens < turn_state.sample_params.max_tokens:
                        turn_state.sample_params = turn_state.sample_params.model_copy(
                            update={"max_tokens": remaining_tokens}
                        )
                image_data = prompt_image_data + [item["image_data"] for item in tool_images]
                if image_data:
                    turn_state.extra_fields["image_data"] = image_data
                assert self.rollout_ctl is not None
                turn_state = await trace_rollout_remote(self.rollout_ctl.generate, turn_state)
                if turn_state.status != Status.COMPLETED:
                    return self._fail(
                        state, f"turn {turn} generate failed: {turn_state.error_msg or turn_state.status}"
                    )
                last_finish_reason = turn_state.finish_reason or "stop"
                if last_finish_reason == "length":
                    n_truncated_turns += 1
                gen_ids = list(turn_state.response_ids or [])
                gen_logprobs = list(turn_state.logprobs or [0.0] * len(gen_ids))
                if len(gen_logprobs) != len(gen_ids):
                    gen_logprobs = (gen_logprobs + [0.0] * len(gen_ids))[: len(gen_ids)]
                if not gen_ids:
                    return self._fail(state, f"turn {turn}: empty generation")
                # 引擎若在工具轮把停止符一并产出，剥掉（wrapper 会重新补上）
                if gen_ids[-1] == self._stop_id:
                    gen_ids = gen_ids[:-1]
                    gen_logprobs = gen_logprobs[:-1]

                call = None
                parsed_from_model = False
                if (
                    self.strict_think_tool_parse
                    and self._think_start_id is not None
                    and self._think_end_id is not None
                ):
                    in_think = (self._think_start_id in cumulative[-3:]) or (self._think_start_id in gen_ids[:3])
                    if in_think and self._think_end_id not in gen_ids:
                        # <think> 未闭合即撞 max_tokens，内部思考文本中的 <tool_call> 字面片段不作为真实调用执行
                        call = None
                        post_think_start = len(gen_ids)
                    elif self._think_end_id in gen_ids:
                        post_think_start = gen_ids.index(self._think_end_id) + 1
                        post_think_text = self.tokenizer.decode(gen_ids[post_think_start:], skip_special_tokens=False)
                        call = self._parse(post_think_text)
                        parsed_from_model = call is not None
                    else:
                        post_think_start = 0
                        gen_text_raw = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                        call = self._parse(gen_text_raw)
                        parsed_from_model = call is not None
                    if (
                        parsed_from_model
                        and self._tool_end_id is not None
                        and self._tool_end_id in gen_ids[post_think_start:]
                    ):
                        # 闭合首个 </tool_call> 后未发 <|im_end|> 继续续写时，截齐到首个 </tool_call>
                        cut = post_think_start + gen_ids[post_think_start:].index(self._tool_end_id) + 1
                        gen_ids = gen_ids[:cut]
                        gen_logprobs = gen_logprobs[:cut]
                else:
                    gen_text_raw = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                    call = self._parse(gen_text_raw)
                    parsed_from_model = call is not None

                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                appended.extend(gen_ids)
                labels_tail.extend(gen_ids)
                logprobs_tail.extend(gen_logprobs)
                tail_mm_flags.extend([0] * len(gen_ids))
                cumulative.extend(gen_ids)

                if call is None and forced_queue:
                    call = forced_queue.pop(0)
                    n_forced += 1
                    logger.info(
                        f"[MultiTurnToolAgentLoop] rollout_id={state.rollout_id} turn {turn}: "
                        f"no parseable tool call, force-injecting {call[0]}({call[1]})"
                    )
                if call is None:
                    final_text = gen_text
                    break
                if turn == self.max_turns:
                    # 末轮仍发出工具调用则直接失败返回，不再白跑一次无消费者的 PDF 渲染
                    break
                name, args = call
                n_tool_calls += 1
                result = tools.call(name, args)
                tool_text = result.text
                if self.final_turn_hint and turn == self.max_turns - 1:
                    tool_text = tool_text + FINAL_TURN_WRAPUP_NOTE
                if result.image_data is not None:
                    pixel_values, grid, num_tokens = self._process_tool_image(result.image_data)
                    prefix_ids = self.tokenizer.encode(
                        self._fmt.img_wrap_prefix.format(tool_text), add_special_tokens=False
                    )
                    suffix_ids = self.tokenizer.encode(self._fmt.img_wrap_suffix, add_special_tokens=False)
                    # train 空间就地展开 N 个 image_token_id；rollout 空间保持单占位符由引擎展开
                    wrapper_ids = prefix_ids + [self._image_token_id] * num_tokens + suffix_ids
                    rollout_wrapper_ids = prefix_ids + [self._image_token_id] + suffix_ids
                    if self.max_total_tokens is not None and (
                        len(train_prompt) + len(appended) + len(wrapper_ids) + _MIN_TURN_TOKENS > self.max_total_tokens
                    ):
                        return self._fail(
                            state,
                            f"turn {turn}: tool wrapper ({len(wrapper_ids)} tokens) exceeds max_total_tokens {self.max_total_tokens}",
                        )
                    tail_pixels.append(pixel_values)
                    tail_grids.append(grid)
                    tail_num_tokens.append(num_tokens)
                    tool_images.append({"page": args.get("page"), "image_data": result.image_data})
                    tail_mm_flags.extend([0] * len(prefix_ids) + [1] * num_tokens + [0] * len(suffix_ids))
                else:
                    wrapper_ids = self.tokenizer.encode(
                        self._fmt.tool_wrap.format(tool_text), add_special_tokens=False
                    )
                    rollout_wrapper_ids = wrapper_ids
                    if self.max_total_tokens is not None and (
                        len(train_prompt) + len(appended) + len(wrapper_ids) + _MIN_TURN_TOKENS > self.max_total_tokens
                    ):
                        return self._fail(
                            state,
                            f"turn {turn}: tool wrapper ({len(wrapper_ids)} tokens) exceeds max_total_tokens {self.max_total_tokens}",
                        )
                    tail_mm_flags.extend([0] * len(wrapper_ids))
                appended.extend(wrapper_ids)
                labels_tail.extend([-100] * len(wrapper_ids))
                logprobs_tail.extend([0.0] * len(wrapper_ids))
                cumulative.extend(rollout_wrapper_ids)

            if not final_text:
                return self._fail(state, f"no final answer within {self.max_turns} turns")

            input_ids = list(train_prompt) + appended
            labels = [-100] * len(train_prompt) + labels_tail
            logprobs = [0.0] * len(train_prompt) + logprobs_tail
            state.input_ids = input_ids
            state.labels = labels
            state.prompt_ids = list(train_prompt)
            state.response_ids = list(appended)
            state.logprobs = logprobs
            state.response = final_text
            state.status = Status.COMPLETED
            state.finish_reason = last_finish_reason
            state.extra_fields["agent_turns"] = turn
            state.extra_fields["agent_tool_calls"] = n_tool_calls
            if n_truncated_turns:
                state.extra_fields["agent_truncated_turns"] = n_truncated_turns
            if n_forced:
                state.extra_fields["agent_forced_tool_calls"] = n_forced
            if tool_images:
                state.extra_fields["agent_tool_images"] = tool_images
                self._extend_mm_info(state, tail_mm_flags, tail_pixels, tail_grids, tail_num_tokens)
            return state
        except Exception as exc:
            logger.warning(
                f"[MultiTurnToolAgentLoop] rollout_id={state.rollout_id} traceback:\n{traceback.format_exc()}"
            )
            return self._fail(state, f"{type(exc).__name__}: {exc}")
        finally:
            tools.close()
