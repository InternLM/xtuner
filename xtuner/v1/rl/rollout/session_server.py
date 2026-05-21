import json
from functools import reduce
from operator import add
from typing import Any, Optional

import numpy as np
import ray
from aiohttp import ClientSession, web
from transformers import AutoTokenizer

from xtuner.v1.utils import get_logger

from .trace_store import TokenizedSegment, get_store


class SessionServer:
    """SessionServer intercepts and records requests sent to a remote LLM API worker.

    It acts as a reverse-proxy (or interceptor) in front of an already running
    worker (like lmdeploy, sglang, or vllm). It binds to a specific (host, port)
    and relays any received traffic to the actual worker URL.

    You can optionally provide before_request and after_response hooks to
    perform extra logging, trace state mutations, or message cleanup before/after
    routing the request to the worker backend.

    Args:
        worker_base_url (str): The base URL of the real worker (e.g. "http://127.0.0.1:8000")
        tokenizer_path (str): The path to the tokenizer model.
        host (str): Host for this session server to listen on.
        port (int): Port for this session server to listen on.
        read_bufsize (int): Buffer limit for line reader in ClientSession. Default is 64MB (2**26).
    """

    def __init__(
        self,
        worker_base_url: str,
        tokenizer_path: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        read_bufsize: int = 2**26,
    ):
        self.worker_base_url = worker_base_url.rstrip("/")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.host = host
        self.port = port
        self.read_bufsize = read_bufsize
        self.store = get_store()
        self.stop_word = self.tokenizer.eos_token or ""

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._lmdeploy_actor: Optional[ray.actor.ActorHandle] = None

    async def on_request(self, req_body: dict) -> dict:
        """Hook for processing/modifying the request before forwarding."""

        session_id = req_body["session_id"]
        # 1. chat_template render 出完整 prompt string，不 tokenize 全量
        prompt_text = self.tokenizer.apply_chat_template(
            req_body["messages"], tools=req_body.get("tools", None), add_generation_prompt=True, tokenize=False
        )

        # 2. Store 做 string prefix match。
        prefix, nodes = await self.store.search.remote(session_id, prompt_text, filter_none=True)
        if prefix:
            get_logger().info(f'Hit prefix cache for session {session_id}')
        delta, delta_ids = prompt_text[len(prefix) :], []
        if delta:
            delta_ids = self.tokenizer.encode(delta, add_special_tokens=False)
            await self.store.insert.remote(session_id, prompt_text, TokenizedSegment(text=delta, token_ids=delta_ids))
        input_ids = reduce(add, [node.value.token_ids for node in nodes] + [delta_ids])

        # 3. 组装 OpenAI chat completions 请求。
        worker_req = {
            **{k: v for k, v in req_body.items() if k not in ['session_id', 'messages']},
            'messages': [],
            'input_ids': input_ids,
            'return_token_ids': True,
            'return_routed_experts': True,
            'logprobs': True,
            'include_stop_str_in_output': True,
        }
        return worker_req

    async def on_response(self, worker_resp: dict) -> dict:
        """Hook for processing the parsed response received from the worker."""

        session_id = worker_resp["session_id"]
        messages = worker_resp['messages']
        tools = worker_resp['tools']
        choice = worker_resp["choices"][0]

        output_token_ids = choice['output_ids']  # len = N_out
        if choice.get("logprobs") and choice["logprobs"].get("content"):
            output_logprobs = [item.get("logprob", 0.0) for item in choice["logprobs"]["content"]]
        else:
            output_logprobs = [0.0] * len(output_token_ids)
        raw_routed_expert = choice.get("routed_experts")  # 本次 call 的 raw routed_expert，可为 None

        # 2. Store 把 input_delta / assistant_output 两个节点补齐字段。
        old_prompt = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False
        )
        messages.append(choice["message"])
        new_prompt = (
            self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=False)
        ).rstrip()
        assert new_prompt.startswith(old_prompt) and new_prompt.endswith(self.stop_word)

        if raw_routed_expert is not None:
            raw_routed_expert = await self._decode_routed_experts(raw_routed_expert)
            if len(raw_routed_expert) > 0:
                num_layers = raw_routed_expert.shape[1]
                topk_experts = raw_routed_expert.shape[2]
                dummy_expert = np.full((1, num_layers, topk_experts), -1, dtype=raw_routed_expert.dtype)
                raw_routed_expert = np.concatenate([dummy_expert, raw_routed_expert], axis=0)

            _, nodes = await self.store.search.remote(session_id, old_prompt, filter_none=True)

            # last node in nodes corresponds to the delta inserted in on_request (if any)
            if nodes:
                delta_node_val: TokenizedSegment = nodes[-1].value
                delta_len = delta_node_val.length
                prefix_len = sum(len(n.value.token_ids) for n in nodes[:-1])
                assert prefix_len + delta_len + len(output_token_ids) == len(raw_routed_expert)

                # split raw_routed_expert
                # raw_routed_expert target shape mapping: [prefix_len + delta_len + response_len, ...]
                delta_expert = raw_routed_expert[prefix_len : prefix_len + delta_len]
                response_expert = raw_routed_expert[prefix_len + delta_len :]

                if delta_len > 0:
                    delta_node_val.expert_key = ray.put(delta_expert)
                    # update delta node in store
                    await self.store.insert.remote(session_id, old_prompt, delta_node_val)

                raw_routed_expert = ray.put(response_expert)
            else:
                raw_routed_expert = ray.put(raw_routed_expert)

        await self.store.insert.remote(
            session_id,
            key=new_prompt,
            value=TokenizedSegment(
                text=new_prompt[len(old_prompt) :],
                token_ids=output_token_ids,
                logprobs=output_logprobs,
                labels=output_token_ids,
                expert_key=raw_routed_expert,
                length=len(output_token_ids),
            ),
        )

        # 3. 返回标准 OpenAI response，session_id 由 SessionClient 层再剥
        resp = {k: v for k, v in worker_resp.items() if k != "messages"}
        return resp

    @property
    def url(self) -> str:
        """The bound URL for the SessionServer."""
        return f"http://{self.host}:{self.port}"

    async def start(self):
        """Start the SessionServer proxy application."""
        if self._site is not None:
            return

        self._app = web.Application()
        self._app.router.add_route("*", "/{path:.*}", self._handle_request)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        get_logger().info(f"SessionServer listening on {self.url} (Forwarding to {self.worker_base_url})")

    async def stop(self):
        """Cleanly stop the SessionServer application."""
        if self._runner:
            await self._runner.cleanup()
        self._site = None
        self._runner = None
        self._app = None
        get_logger().info("SessionServer stopped.")

    async def _handle_request(self, request: web.Request) -> web.Response:
        """Proxy handler for the worker API."""

        # Read the request body
        request_body = await request.read()
        request_data = session_id = messages = None
        orig_logprobs = orig_return_token_ids = orig_return_routed_experts = False

        if request_body:
            try:
                request_data = json.loads(request_body)

                orig_logprobs = request_data.get('logprobs', False)
                orig_return_token_ids = request_data.get('return_token_ids', False)
                orig_return_routed_experts = request_data.get('return_routed_experts', False)

                session_id = request_data.get('session_id')
                messages = request_data.get('messages')
                tools = request_data.get('tools', None)

                # Apply purely abstract on_request processing
                request_data = await self.on_request(request_data)
                # Re-serialize the modified payload back to bytes
                request_body = json.dumps(request_data).encode("utf-8")
            except json.JSONDecodeError:
                pass

        # Build forwarding headers, dropping original Host
        forward_headers = dict(request.headers)
        forward_headers.pop("Host", None)
        forward_headers.pop("host", None)
        forward_headers.pop("Content-Length", None)
        forward_headers.pop("content-length", None)

        # Re-build Path
        req_path = request.match_info["path"]
        target_url = f"{self.worker_base_url}/{req_path.lstrip('/')}"
        if request.query_string:
            target_url += f"?{request.query_string}"

        is_stream = request_data.get("stream", False) if request_data else False

        def _clean_data(data: dict) -> bool:
            modified = False
            for key, drop in [
                ("logprobs", not orig_logprobs),
                ("output_ids", not orig_return_token_ids),
                ("routed_experts", not orig_return_routed_experts),
            ]:
                if drop and key in data:
                    data.pop(key)
                    modified = True
                if drop:
                    for c in data.get("choices", []):
                        if key in c:
                            c.pop(key)
                            modified = True

            for c in data.get("choices", []):
                if c.get("message") and isinstance(c["message"].get("content"), str):
                    if self.stop_word in c["message"]["content"]:
                        c["message"]["content"] = c["message"]["content"].replace(self.stop_word, "")
                        modified = True
                if c.get("delta") and isinstance(c["delta"].get("content"), str):
                    if self.stop_word in c["delta"]["content"]:
                        c["delta"]["content"] = c["delta"]["content"].replace(self.stop_word, "")
                        modified = True

            return modified

        # Forward the request to the upstream worker
        # read_bufsize controls StreamReader's line buffer limit; SSE events with large
        # tool_calls/reasoning_content payloads can exceed the 64KB default and trigger
        # "Chunk too big" from readuntil(b"\n").
        async with ClientSession(read_bufsize=self.read_bufsize) as client:
            async with client.request(
                method=request.method, url=target_url, headers=forward_headers, data=request_body
            ) as resp:

                # Setup proper stream vs sync response objects
                if is_stream:
                    response_chunks = []
                    response = web.StreamResponse(
                        status=resp.status,
                        headers={
                            k: v
                            for k, v in resp.headers.items()
                            if k.lower() not in ("transfer-encoding", "content-length", "content-encoding")
                        },
                    )
                    await response.prepare(request)
                    async for line in resp.content:
                        # Keep unmodified line for trace store parsing
                        response_chunks.append(line)

                        # Dynamically prune added fields before writing to client
                        if request_data is not None and line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                            try:
                                text = line.decode("utf-8")
                                data = json.loads(text[6:])
                                if _clean_data(data):
                                    line = ("data: " + json.dumps(data) + "\n").encode("utf-8")
                            except Exception:
                                pass

                        await response.write(line)
                    await response.write_eof()
                    raw_response = b"".join(response_chunks)  # Original raw response for exact tracing
                else:
                    raw_response = await resp.read()
                    final_raw_response = raw_response

                    if request_data is not None:
                        try:
                            clean_data = json.loads(raw_response)
                            if _clean_data(clean_data):
                                final_raw_response = json.dumps(clean_data).encode("utf-8")
                        except Exception:
                            pass

                    response = web.Response(
                        status=resp.status,
                        headers={
                            k: v
                            for k, v in resp.headers.items()
                            if k.lower() not in ("transfer-encoding", "content-length", "content-encoding")
                        },
                        body=final_raw_response,  # Modifed raw response without our injected trace params
                    )

        # Apply abstract on_response processing
        response_data = None
        if request_data:
            if is_stream:
                response_data = self._parse_stream_response(raw_response)
            else:
                try:
                    response_data = json.loads(raw_response)
                except json.JSONDecodeError:
                    pass

            if response_data is not None:
                for c in response_data.get("choices", []):
                    if c.get("message") and isinstance(c["message"].get("content"), str):
                        c["message"]["content"] = c["message"]["content"].replace(self.stop_word, "")
                        for tc in c['message'].get('tool_calls') or []:
                            if isinstance(tc.get('function', {}).get('arguments'), str):
                                tc['function']['arguments'] = json.loads(tc['function']['arguments'])

                response_data['session_id'] = session_id
                response_data['messages'] = messages
                response_data['tools'] = tools
                await self.on_response(response_data)

        return response

    async def _decode_routed_experts(self, routed_experts: Any) -> np.ndarray:
        if isinstance(routed_experts, str):
            if self._lmdeploy_actor is None:
                self._lmdeploy_actor = ray.get_actor('shared_store', namespace='lmdeploy')
            assert self._lmdeploy_actor is not None, "LMDeploy actor should be available in the shared store."
            routed_experts_data = await self._lmdeploy_actor.get.remote(routed_experts)
            return np.asarray(routed_experts_data)
        return np.asarray(routed_experts)

    @staticmethod
    def _parse_stream_response(raw: bytes) -> Optional[dict]:
        """Parse SSE stream to reconstruct the complete final message state."""
        text = raw.decode("utf-8", errors="replace")
        events = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        if not events:
            return None

        # Reconstruct standard stream output (Assuming OpenAI format here)
        message = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
        content_parts = []
        tool_calls_map = {}
        usage = {}

        for event in events:
            if event.get("id") and "id" not in message:
                message["id"] = event["id"]
            if event.get("model"):
                message["model"] = event["model"]

            choices = event.get("choices", [])
            for choice in choices:
                delta = choice.get("delta", {})

                # Check text content
                if delta.get("content"):
                    content_parts.append(delta["content"])

                # Check output ids
                if choice.get('output_ids'):
                    assistant_choice = message["choices"][0]
                    if 'output_ids' not in assistant_choice:
                        assistant_choice['output_ids'] = []
                    assistant_choice['output_ids'].extend(choice['output_ids'])

                # Check routed experts
                if choice.get('routed_experts'):
                    assistant_choice = message["choices"][0]
                    if 'routed_experts' not in assistant_choice:
                        assistant_choice['routed_experts'] = []
                    assistant_choice['routed_experts'].extend(choice['routed_experts'])

                # Check logprobs
                if choice.get("logprobs") and choice["logprobs"].get("content"):
                    assistant_choice = message["choices"][0]
                    if "logprobs" not in assistant_choice:
                        assistant_choice["logprobs"] = {"content": []}
                    assistant_choice["logprobs"]["content"].extend(choice["logprobs"]["content"])

                # Check reasoning content
                if delta.get("reasoning_content"):
                    assistant_msg = message["choices"][0]["message"]
                    assistant_msg["reasoning_content"] = (
                        assistant_msg.get("reasoning_content", "") + delta["reasoning_content"]
                    )

                # Check tool calls
                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc_delta.get("id", ""),
                            "type": tc_delta.get("type", "function"),
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_map[idx]
                    fn = tc_delta.get("function", {})
                    if fn.get("name"):
                        tc["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tc["function"]["arguments"] += fn["arguments"]

                if choice.get("finish_reason"):
                    message["choices"][0]["finish_reason"] = choice["finish_reason"]

            if event.get("usage") is not None:
                usage = event["usage"]

        msg = message["choices"][0]["message"]
        msg["content"] = "".join(content_parts)
        if tool_calls_map:
            msg["tool_calls"] = [tool_calls_map[i] for i in sorted(tool_calls_map)]
        if usage:
            message["usage"] = usage

        return message
