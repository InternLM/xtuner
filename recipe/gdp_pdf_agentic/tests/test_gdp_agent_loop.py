#!/usr/bin/env python
"""GdpToolAgentLoop 的 CPU 单测：mock rollout generate，验证多轮流与训练字段拼装。

运行: PYTHONPATH=repo_xtuner:. $PT29_RL/bin/python scripts/test_gdp_agentic_loop.py
"""

import asyncio
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from recipe.gdp_pdf_agentic.agent_loop import GdpToolAgentLoopConfig  # noqa: E402
from xtuner.v1.data_proto.rl_data import RolloutState, Status  # noqa: E402
from xtuner.v1.rl.agent_loop.multi_turn_tool_agent_loop import parse_tool_call  # noqa: E402


MODEL = os.environ.get("GDP_TEST_MODEL", "/mnt/shared-storage-user/llmrazor-share/model/GLM-5.3-Flash-25B")
DATA_ROOT = Path(
    os.environ.get(
        "GDP_DATA_DIR",
        "/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/glm53-multimodal-agentic-rl/data/gdp_pdf",
    )
)
PDF = (
    DATA_ROOT
    / "pdfs/1558afcc-854e-471d-933b-51cd8de0a0dc_b5d91c2a-227e-49e9-a6a0-c47576e333ae_78de1655-d256-4c54-a172-c6a7436f8b84.pdf"
)


class _RemoteMethod:
    """模拟 ray actor 的 .remote() 调用约定。"""

    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return await self._fn(*args, **kwargs)


class FakeRolloutCtl:
    """脚本化两轮生成：第 1 轮工具调用，第 2 轮最终回答。"""

    def __init__(self, tokenizer, tool_call: str = "get_page_text"):
        self.tok = tokenizer
        self.calls = 0
        self.seen_prompts = []
        self.seen_image_data = []
        self.generate = _RemoteMethod(self._generate)
        self._tool_call = tool_call

    async def _generate(self, turn_state):
        self.calls += 1
        self.seen_prompts.append(list(turn_state.tokens))
        self.seen_image_data.append(list(turn_state.extra_fields.get("image_data") or []))
        if self.calls == 1:
            text = f"</think><tool_call>{self._tool_call}<arg_key>page</arg_key><arg_value>1</arg_value></tool_call>"
        else:
            text = "</think>The centered features are p1175, P1022 and a1085."
        ids = self.tok.encode(text, add_special_tokens=False)
        turn_state.response_ids = ids
        turn_state.logprobs = [-0.1 * i for i in range(1, len(ids) + 1)]
        turn_state.status = Status.COMPLETED
        return turn_state


def build_state(tok):
    prompt_ids = tok.encode("[gMASK]<sop><|system|>s<|user|>q<|image|><|assistant|><think>", add_special_tokens=False)
    train_prompt = prompt_ids[:4] + [900000 + i for i in range(3)] + prompt_ids[4:]  # 模拟展开后更长
    return RolloutState(
        message=[{"role": "user", "content": "q"}],
        prompt_ids=prompt_ids,
        num_tokens=len(prompt_ids),
        data_source={"gdp_pdf": 1.0},
        extra_fields={"train_prompt_ids": train_prompt, "pdf_path": str(PDF)},
    )


def test_text_tool_flow():
    # 1. 工具调用解析
    assert parse_tool_call("blah") is None
    name, args = parse_tool_call(
        "</think><tool_call>get_page_text<arg_key>page</arg_key><arg_value>3</arg_value></tool_call>"
    )
    assert name == "get_page_text" and args == {"page": "3"}, (name, args)
    print("parse_tool_call ok")

    # 2. 完整两轮流 + 训练字段
    cfg = GdpToolAgentLoopConfig(hf_checkpoint=MODEL, max_turns=4)
    loop = cfg.build_local(rollout_controller=None)
    tok = loop.tokenizer
    fake = FakeRolloutCtl(tok)
    loop.rollout_ctl = fake
    state = build_state(tok)
    out = asyncio.run(loop.generate_sample(state))

    assert out.status == Status.COMPLETED, out.error_msg
    assert fake.calls == 2, fake.calls
    train_prompt = state.extra_fields["train_prompt_ids"]
    # prompt/response 切分
    assert out.prompt_ids == list(train_prompt)
    assert out.input_ids == list(train_prompt) + out.response_ids
    # labels：prompt 与 wrapper 段 -100，两个生成段被监督
    gen1_len = len(
        fake.tok.encode(
            "</think><tool_call>get_page_text<arg_key>page</arg_key><arg_value>1</arg_value></tool_call>",
            add_special_tokens=False,
        )
    )
    resp = out.response_ids
    lbl = out.labels[len(train_prompt) :]
    assert lbl[:gen1_len] == resp[:gen1_len], "gen1 span not supervised"
    wrap_start = gen1_len
    wrap_len = 0
    for i, (tok_id, lb) in enumerate(zip(resp[wrap_start:], lbl[wrap_start:])):
        if lb == -100:
            wrap_len += 1
        else:
            break
    assert wrap_len > 0, "no wrapper span"
    assert set(lbl[wrap_start : wrap_start + wrap_len]) == {-100}
    gen2 = resp[wrap_start + wrap_len :]
    assert lbl[wrap_start + wrap_len :] == gen2, "gen2 span not supervised"
    # wrapper 解码含工具结果
    wrap_text = tok.decode(resp[wrap_start : wrap_start + wrap_len])
    assert wrap_text.startswith("<|observation|><tool_response>") and "1558" not in wrap_text
    assert "<|assistant|><think>" in wrap_text
    # logprobs 全长对齐、非生成位为 0
    assert len(out.logprobs) == len(out.input_ids)
    assert all(lp == 0.0 for lp in out.logprobs[: len(train_prompt)])
    assert all(
        lp == 0.0 for lp in out.logprobs[len(train_prompt) + wrap_start : len(train_prompt) + wrap_start + wrap_len]
    )
    # 第二轮收到的 prompt = 第一轮 prompt + gen1 + wrapper（rollout 空间）
    p0, p1 = fake.seen_prompts
    assert p1[: len(p0)] == p0, "turn-2 prompt must extend turn-1 prompt"
    assert len(p1) == len(p0) + gen1_len + wrap_len
    assert out.response == "</think>The centered features are p1175, P1022 and a1085."
    assert out.extra_fields["agent_turns"] == 2 and out.extra_fields["agent_tool_calls"] == 1
    print("two-turn flow + training fields ok")
    print(
        f"  input_ids={len(out.input_ids)} prompt={len(train_prompt)} gen1={gen1_len} wrap={wrap_len} gen2={len(gen2)}"
    )


def build_image_state(tok, feat_dim: int):
    prompt_ids = tok.encode("[gMASK]<sop><|system|>s<|user|>q<|image|><|assistant|><think>", add_special_tokens=False)
    train_prompt = prompt_ids[:4] + [900000 + i for i in range(3)] + prompt_ids[4:]  # 模拟展开后更长
    prompt_flags = np.zeros(len(train_prompt), dtype=np.int64)
    prompt_flags[4:7] = 1  # prompt 区图片段
    mm_info = {
        "num_img_tokens": [3],
        "pixel_values": np.zeros((12, feat_dim), dtype=np.float32),  # 3 token * merge_unit=4 patch
        "image_grid_thw": np.array([[1, 2, 6]]),  # prod=12, //4 = 3 token
        "mm_token_type_ids": prompt_flags[None, :],
    }
    return RolloutState(
        message=[{"role": "user", "content": "q"}],
        prompt_ids=prompt_ids,
        num_tokens=len(prompt_ids),
        data_source={"gdp_pdf": 1.0},
        mm_info=mm_info,
        extra_fields={
            "train_prompt_ids": train_prompt,
            "pdf_path": str(PDF),
            "image_data": ["data:image/png;base64,cHJvbXB0LWltZw=="],
        },
    )


def test_image_tool_flow():
    cfg = GdpToolAgentLoopConfig(hf_checkpoint=MODEL, max_turns=4)
    loop = cfg.build_local(rollout_controller=None)
    tok = loop.tokenizer
    # 用一张小图探测 processor 的像素特征维度，供构造假 prompt 媒体字段
    import base64 as b64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 255, 255)).save(buf, format="PNG")
    probe_uri = "data:image/png;base64," + b64.b64encode(buf.getvalue()).decode()
    probe_pixels, _, _ = loop._process_tool_image(probe_uri)
    fake = FakeRolloutCtl(tok, tool_call="get_page_image")
    loop.rollout_ctl = fake
    state = build_image_state(tok, feat_dim=probe_pixels.shape[1])
    out = asyncio.run(loop.generate_sample(state))

    assert out.status == Status.COMPLETED, out.error_msg
    assert fake.calls == 2, fake.calls
    train_prompt = state.extra_fields["train_prompt_ids"]
    n_prompt = len(train_prompt)
    image_token_id = loop._image_token_id

    # 工具图片经 processor 展开的 token 数
    n_img = out.mm_info["num_img_tokens"][-1]
    assert n_img > 0
    resp = out.response_ids
    lbl = out.labels[n_prompt:]
    img_positions = [i for i, t in enumerate(resp) if t == image_token_id]
    assert len(img_positions) == n_img, f"train space must expand placeholder to {n_img} tokens"
    first, last = img_positions[0], img_positions[-1]
    assert last - first + 1 == n_img, "image tokens must be contiguous"
    # 图片段不监督、logprobs 为 0
    assert set(lbl[first : last + 1]) == {-100}
    assert all(lp == 0.0 for lp in out.logprobs[n_prompt + first : n_prompt + last + 1])
    # wrapper 文本部分解码含工具说明与边界
    wrap_text = tok.decode(resp[:first]) + tok.decode(resp[last + 1 :])
    assert "<|observation|><tool_response>" in wrap_text and "rendered as" in wrap_text
    assert "<|end_of_image|></tool_response><|assistant|><think>" in wrap_text
    # rollout 空间第二轮 prompt 只含单占位符
    p0, p1 = fake.seen_prompts
    assert p1[: len(p0)] == p0
    assert list(p1).count(image_token_id) == list(p0).count(image_token_id) + 1
    # rollout 请求 image_data = prompt 图 + 工具图（按序）
    assert len(fake.seen_image_data[0]) == 1
    assert len(fake.seen_image_data[1]) == 2
    assert fake.seen_image_data[1][-1].startswith("data:image/png;base64,")
    # mm_info：num_img_tokens 追加、pixel/grid 按序拼接、mm_token_type_ids 全序列覆盖
    assert out.mm_info["num_img_tokens"] == [3, n_img]
    flags = np.asarray(out.mm_info["mm_token_type_ids"])
    # TrainBatch 训练输入 input_ids[:-1]，flags 相应去末位（末位 flag 必为 0）
    assert flags.shape == (1, len(out.input_ids) - 1)
    assert flags[0, 4:7].tolist() == [1, 1, 1]  # prompt 区保留
    tail_flags = flags[0, n_prompt:]
    assert tail_flags[first : last + 1].tolist() == [1] * n_img
    assert tail_flags.sum() == n_img  # tail 只有工具图一段为 1
    assert flags.sum() == 3 + n_img  # 全序列：prompt 3 + 工具图 n_img
    n_patch_new = int(np.prod(out.mm_info["image_grid_thw"][-1]))
    assert out.mm_info["pixel_values"].shape[0] == 12 + n_patch_new
    assert out.extra_fields["agent_tool_images"][0]["page"] == "1"
    print("image-tool flow ok")
    print(f"  n_img={n_img} seq={len(out.input_ids)} flags_sum={int(flags.sum())}")


class GarbageRolloutCtl:
    """模拟减层 policy：两轮都输出不可解析的乱码长文本。"""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.calls = 0
        self.generate = _RemoteMethod(self._generate)

    async def _generate(self, turn_state):
        self.calls += 1
        ids = self.tok.encode(f"</think>garbage rambling {self.calls} " + "blah " * 20, add_special_tokens=False)
        turn_state.response_ids = ids
        turn_state.logprobs = [0.0] * len(ids)
        turn_state.status = Status.COMPLETED
        return turn_state


def test_forced_tool_call_flow():
    from xtuner.v1.rl.agent_loop.multi_turn_tool_agent_loop import _parse_forced_call

    assert _parse_forced_call("get_page_image(page=1)") == ("get_page_image", {"page": "1"})
    cfg = GdpToolAgentLoopConfig(hf_checkpoint=MODEL, max_turns=4, force_tool_calls=["get_page_image(page=1)"])
    loop = cfg.build_local(rollout_controller=None)
    tok = loop.tokenizer
    import base64 as b64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 255, 255)).save(buf, format="PNG")
    probe_uri = "data:image/png;base64," + b64.b64encode(buf.getvalue()).decode()
    probe_pixels, _, _ = loop._process_tool_image(probe_uri)
    loop.rollout_ctl = GarbageRolloutCtl(tok)
    state = build_image_state(tok, feat_dim=probe_pixels.shape[1])
    out = asyncio.run(loop.generate_sample(state))

    assert out.status == Status.COMPLETED, out.error_msg
    assert out.extra_fields["agent_forced_tool_calls"] == 1
    assert out.extra_fields["agent_tool_calls"] == 1
    assert out.extra_fields["agent_turns"] == 2
    n_img = out.mm_info["num_img_tokens"][-1]
    resp = out.response_ids
    assert resp.count(loop._image_token_id) == n_img
    assert out.extra_fields["agent_tool_images"][0]["page"] == "1"
    flags = np.asarray(out.mm_info["mm_token_type_ids"])
    assert flags.shape == (1, len(out.input_ids) - 1)
    assert flags.sum() == 3 + n_img
    print("forced tool-call flow ok")


QWEN_MODEL = "/mnt/shared-storage-user/llmrazor-share/model/Qwen3.5-9B"


def test_qwen_flow():
    from xtuner.v1.rl.agent_loop.multi_turn_tool_agent_loop import parse_qwen_tool_call

    assert parse_qwen_tool_call(
        'blah <tool_call>\n{"name": "get_page_image", "arguments": {"page": 2}}\n</tool_call>'
    ) == ("get_page_image", {"page": "2"})
    assert parse_qwen_tool_call("<tool_call>not json</tool_call>") is None

    cfg = GdpToolAgentLoopConfig(hf_checkpoint=QWEN_MODEL, max_turns=4, model_family="qwen35_vl")
    loop = cfg.build_local(rollout_controller=None)
    tok = loop.tokenizer
    import base64 as b64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 255, 255)).save(buf, format="PNG")
    probe_uri = "data:image/png;base64," + b64.b64encode(buf.getvalue()).decode()
    probe_pixels, _, _ = loop._process_tool_image(probe_uri)
    feat_dim = probe_pixels.shape[1]

    class QwenFakeRollout:
        def __init__(self):
            self.calls = 0
            self.generate = _RemoteMethod(self._generate)

        async def _generate(self, turn_state):
            self.calls += 1
            if self.calls == 1:
                text = '<tool_call>\n{"name": "get_page_image", "arguments": {"page": 1}}\n</tool_call><|im_end|>'
            else:
                text = "The answer is p1175.<|im_end|>"
            ids = tok.encode(text, add_special_tokens=False)
            turn_state.response_ids = ids
            turn_state.logprobs = [0.0] * len(ids)
            turn_state.status = Status.COMPLETED
            return turn_state

    fake = QwenFakeRollout()
    loop.rollout_ctl = fake
    image_token_id = loop._image_token_id
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    prompt_ids = tok.encode(
        "<|im_start|>user\nq<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    train_prompt = [i if i != image_token_id else -1 for i in prompt_ids]
    expand_at = train_prompt.index(-1)
    train_prompt = train_prompt[:expand_at] + [image_token_id] * 3 + train_prompt[expand_at + 1 :]
    mm_info = {
        "num_img_tokens": [3],
        "pixel_values": np.zeros((12, feat_dim), dtype=np.float32),
        "image_grid_thw": np.array([[1, 2, 6]]),
    }
    state = RolloutState(
        message=[{"role": "user", "content": "q"}],
        prompt_ids=prompt_ids,
        num_tokens=len(prompt_ids),
        data_source={"gdp_pdf": 1.0},
        mm_info=mm_info,
        extra_fields={
            "train_prompt_ids": train_prompt,
            "pdf_path": str(PDF),
            "image_data": ["data:image/png;base64,cHJvbXB0LWltZw=="],
        },
    )
    out = asyncio.run(loop.generate_sample(state))

    assert out.status == Status.COMPLETED, out.error_msg
    assert out.extra_fields["agent_tool_calls"] == 1 and out.extra_fields["agent_turns"] == 2
    n_img = out.mm_info["num_img_tokens"][-1]
    assert out.mm_info["num_img_tokens"] == [3, n_img]
    assert "mm_token_type_ids" not in out.mm_info  # qwen 不需要模态标记
    resp = out.response_ids
    assert resp.count(image_token_id) == n_img  # train 空间工具图展开
    # <|im_end|> 被剥掉且 wrapper 重补：解码含 tool_response 边界
    wrap_text = tok.decode(resp)
    assert "<tool_response>" in wrap_text and "rendered as" in wrap_text
    assert "<|vision_start|>" in wrap_text and "<|im_start|>assistant" in wrap_text
    # 剥掉停止符后没有连续两个 im_end
    ids = resp
    assert not any(a == im_end_id == b for a, b in zip(ids, ids[1:]))
    print("qwen profile flow ok")


def test_long_window_guards():
    from xtuner.v1.rl.agent_loop.multi_turn_tool_agent_loop import FINAL_TURN_WRAPUP_NOTE

    from xtuner.v1.data_proto.rl_data import SampleParams

    cfg = GdpToolAgentLoopConfig(
        hf_checkpoint=QWEN_MODEL,
        max_turns=3,
        model_family="qwen35_vl",
        sample_params=SampleParams(max_tokens=4096),
        max_total_tokens=2000,
        final_turn_hint=True,
        strict_think_tool_parse=True,
    )
    loop = cfg.build_local(rollout_controller=None)
    tok = loop.tokenizer
    prompt_ids = tok.encode(
        "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n",
        add_special_tokens=False,
    )

    # Case A: <think> 未闭合撞 length，内部出现的 <tool_call> 字面片段不得被误判为真实工具调用
    class UnclosedThinkRollout:
        def __init__(self):
            self.calls = 0
            self.generate = _RemoteMethod(self._generate)

        async def _generate(self, turn_state):
            self.calls += 1
            text = 'maybe call <tool_call>{"name": "get_doc_meta", "arguments": {}}</tool_call> still thinking...'
            ids = tok.encode(text, add_special_tokens=False)
            turn_state.response_ids = ids
            turn_state.logprobs = [-0.1] * len(ids)
            turn_state.finish_reason = "length"
            turn_state.status = Status.COMPLETED
            return turn_state

    loop.rollout_ctl = UnclosedThinkRollout()
    st_a = RolloutState(
        message=[{"role": "user", "content": "q"}],
        prompt_ids=prompt_ids,
        num_tokens=len(prompt_ids),
        data_source={"gdp_pdf": 1.0},
        extra_fields={"train_prompt_ids": list(prompt_ids), "pdf_path": str(PDF)},
    )
    out_a = asyncio.run(loop.generate_sample(st_a))
    assert out_a.status == Status.COMPLETED
    assert out_a.finish_reason == "length"
    assert out_a.extra_fields["agent_tool_calls"] == 0
    assert out_a.extra_fields["agent_turns"] == 1
    assert out_a.extra_fields["agent_truncated_turns"] == 1

    # Case B: </think> 后首个 </tool_call> 之后多余续写被截齐；第 max_turns-1 轮注入收尾提示；动态扣减 max_tokens
    class MultiTurnGuardRollout:
        def __init__(self):
            self.calls = 0
            self.seen_max_tokens = []
            self.seen_prompts = []
            self.generate = _RemoteMethod(self._generate)

        async def _generate(self, turn_state):
            self.calls += 1
            self.seen_max_tokens.append(turn_state.sample_params.max_tokens)
            self.seen_prompts.append(list(turn_state.tokens))
            if self.calls == 1:
                text = (
                    'plan <tool_call>{"name": "get_page_text", "arguments": {"page": 99}}</tool_call></think>\n'
                    '<tool_call>{"name": "get_doc_meta", "arguments": {}}</tool_call>\n'
                    "hallucinated trailing tokens that must be trimmed"
                )
                finish = "length"
            elif self.calls == 2:
                text = '<think>\nneed page 1\n</think>\n<tool_call>{"name": "get_page_text", "arguments": {"page": 1}}</tool_call><|im_end|>'
                finish = "stop"
            else:
                text = "<think>\ndone\n</think>\nFinal answer: centered features p1175.<|im_end|>"
                finish = "stop"
            ids = tok.encode(text, add_special_tokens=False)
            turn_state.response_ids = ids
            turn_state.logprobs = [-0.2] * len(ids)
            turn_state.finish_reason = finish
            turn_state.status = Status.COMPLETED
            return turn_state

    guard_fake = MultiTurnGuardRollout()
    loop.rollout_ctl = guard_fake
    st_b = RolloutState(
        message=[{"role": "user", "content": "q"}],
        prompt_ids=prompt_ids,
        num_tokens=len(prompt_ids),
        tools=[{"type": "function", "function": {"name": "get_doc_meta"}}],
        data_source={"gdp_pdf": 1.0},
        extra_fields={"train_prompt_ids": list(prompt_ids), "pdf_path": str(PDF)},
    )
    out_b = asyncio.run(loop.generate_sample(st_b))
    assert out_b.status == Status.COMPLETED, out_b.error_msg
    assert out_b.finish_reason == "stop"
    assert out_b.extra_fields["agent_turns"] == 3
    assert out_b.extra_fields["agent_tool_calls"] == 2
    decoded_full = tok.decode(out_b.response_ids, skip_special_tokens=False)
    # 1) <think> 内的 page=99 假调用未执行，真正执行的是 </think> 后的 get_doc_meta
    assert '{"n_pages":' in decoded_full
    # 2) 首个 </tool_call> 后的垃圾续写已被截除
    assert "hallucinated trailing tokens" not in decoded_full
    # 3) turn 2（即 max_turns-1）的 wrapper 注入了收尾提示，驱动 turn 3 直接给答案
    assert FINAL_TURN_WRAPUP_NOTE.strip() in decoded_full
    # 4) 每轮 sample_params.max_tokens 严格随已用长度单调递减，总序列不超 max_total_tokens
    assert guard_fake.seen_max_tokens[0] == 2000 - len(prompt_ids)
    assert guard_fake.seen_max_tokens[2] < guard_fake.seen_max_tokens[1] < guard_fake.seen_max_tokens[0]
    assert len(out_b.input_ids) <= 2000

    # Case C: 预算不足时优雅返回 Status.FAILED，不产生超 pack_max_length 的样本
    cfg_tight = GdpToolAgentLoopConfig(
        hf_checkpoint=QWEN_MODEL,
        max_turns=3,
        model_family="qwen35_vl",
        max_total_tokens=len(prompt_ids) + 10,
    )
    loop_tight = cfg_tight.build_local(rollout_controller=None)
    loop_tight.rollout_ctl = guard_fake
    st_c = RolloutState(
        message=[{"role": "user", "content": "q"}],
        prompt_ids=prompt_ids,
        num_tokens=len(prompt_ids),
        data_source={"gdp_pdf": 1.0},
        extra_fields={"train_prompt_ids": list(prompt_ids), "pdf_path": str(PDF)},
    )
    out_c = asyncio.run(loop_tight.generate_sample(st_c))
    assert out_c.status == Status.FAILED and "max_total_tokens" in (out_c.error_msg or "")
    print("long-window guards flow ok")


if __name__ == "__main__":
    test_text_tool_flow()
    test_image_tool_flow()
    test_forced_tool_call_flow()
    test_qwen_flow()
    test_long_window_guards()
