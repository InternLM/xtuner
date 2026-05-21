import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import ray
from pydantic import BaseModel, Field, StrictStr

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.utils import get_logger


_STORE_NAME = "rollout_trace_store"
_handle_cache: Any = None


class TraceState(str, Enum):
    ROLLOUT_RUNNING = "RolloutRunning"
    ROLLOUT_FINISHED = "RolloutFinished"
    TRAIN_RUNNING = "TrainRunning"
    TRAIN_FINISHED = "TrainFinished"
    TO_BE_RELEASED = "ToBeReleased"
    RELEASED = "Released"


_ROLLOUT_RELEASE_STATUSES = {Status.FAILED, Status.FILTERED, Status.EXPIRED}
_ROLLOUT_KEEP_RUNNING_STATUSES = {
    Status.ABORTED,
    Status.INIT,
}


def _free_ray_refs(obj: Any):
    """Recursively free ray.ObjectRef instances trapped inside an object.

    Args:
        obj (Any): The object that may contain ray.ObjectRef references (e.g., dict, list, tuple).
    """
    if isinstance(obj, ray.ObjectRef):
        try:
            ray.internal.free([obj], local_only=False)
        except Exception as e:
            get_logger().error(f"Failed to free Ray ObjectRef {obj}: {e}")
    elif isinstance(obj, dict):
        for v in obj.values():
            _free_ray_refs(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _free_ray_refs(v)
    elif hasattr(obj, "model_dump"):  # Pydantic v2
        for v in obj.model_dump().values() if hasattr(obj.model_dump, "__call__") else {}.values():
            _free_ray_refs(v)
    elif hasattr(obj, "dict") and callable(getattr(obj, "dict")):  # Pydantic v1
        for v in obj.dict().values():
            _free_ray_refs(v)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            _free_ray_refs(v)


def make_expert_key(session_id: str) -> str:
    """Build a stable key for a routed experts object."""
    return f"{session_id}:routed_experts"


class TokenizedSegment(BaseModel):
    text: str
    token_ids: List[int]
    labels: List[int] | None = Field(default=None, repr=False)
    logprobs: List[float] | None = Field(default=None, repr=False)
    expert_key: StrictStr | None = Field(default=None, repr=False)
    length: int | None = None

    def model_post_init(self, _):
        if self.labels is None:
            self.labels = [-100] * len(self.token_ids)
        if self.logprobs is None:
            self.logprobs = [0.0] * len(self.token_ids)
        if self.length is None:
            self.length = len(self.token_ids)
        assert len(self.token_ids) == len(self.labels) == len(self.logprobs)


@dataclass
class TreeNode:
    """A node in the prefix tree (Trie).

    Attributes:
        value (Optional[Any]): The value stored in the node.
        parent (Optional["TreeNode"]): The parent node.
        children (dict[str, "TreeNode"]): The child nodes, mapping string tokens to TreeNodes.
        created_at (float): The timestamp when the node was created.
    """

    value: Optional[Any]
    parent: Optional["TreeNode"]
    children: dict[str, "TreeNode"] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class Trie:
    def __init__(self):
        """Initialize the prefix tree (Trie)."""
        self.root = TreeNode(value=None, parent=None)
        self.state = TraceState.ROLLOUT_RUNNING
        self.expert_key: str | None = None
        self.updated_at = time.time()

    def touch(self) -> None:
        """Record that this session was updated."""
        self.updated_at = time.time()

    def keys(self) -> List[str]:
        """Get all keys (i.e., strings) stored in the Trie."""
        result = []

        def _collect(node: TreeNode, path: List[str]):
            if node.value is not None:
                result.append("".join(path))
            for token, child in node.children.items():
                _collect(child, path + [token])

        _collect(self.root, [])
        return result

    def insert(self, key: str, value: Any) -> None:
        """Insert a (key, value) pair into the Trie.

        Args:
            key (str): The key string to insert.
            value (Any): The value to store at the destination node.
        """
        node = self.root
        while key:
            best_prefix = ""
            for token in node.children:
                if key.startswith(token) and len(token) > len(best_prefix):
                    best_prefix = token

            if best_prefix:
                node = node.children[best_prefix]
                key = key[len(best_prefix) :]
            else:
                # create a new child node for the remaining key
                node.children[key] = TreeNode(value=None, parent=node)
                node = node.children[key]
                break

        node.value = value
        self.touch()

    def search(self, text: str, filter_none: bool = False) -> Tuple[str, List["TreeNode"]]:
        """Search for the longest prefix matching the input text.

        Args:
            text (str): The input string to search for.
            filter_none (bool): If True, only returns nodes whose value is not None
                (i.e., nodes where a value was explicitly inserted).

        Returns:
            Tuple[str, List["TreeNode"]]: A tuple containing the matched longest prefix string
            and a list of nodes along the matched path (excluding the root).
        """
        node = self.root
        key_path, matched_nodes = [], []

        while text:
            best_prefix = ""
            for token in node.children:
                if text.startswith(token) and len(token) > len(best_prefix):
                    best_prefix = token

            if best_prefix:
                node = node.children[best_prefix]
                matched_nodes.append(node)
                key_path.append(best_prefix)
                text = text[len(best_prefix) :]
            else:
                break

        if filter_none:
            matched_nodes = [n for n in matched_nodes if n.value is not None]

        return "".join(key_path), matched_nodes

    def release(self, key: str | None = None):
        """Release nodes and free associated resources (e.g. Ray objects) for a
        given key.

        If the key is None, the entire Trie is released. Prunes empty branches bottom-up.

        Args:
            key (Optional[str]): The key string whose associated nodes should be released.
                If None, releases the entire tree.
        """

        def _free_subtree(node: TreeNode):
            for child in node.children.values():
                _free_subtree(child)
            if node.value is not None:
                _free_ray_refs(node.value)
                node.value = None
            node.children.clear()

        if key is None:
            _free_subtree(self.root)
            self.touch()
            return

        node = self.root
        path = []
        while key:
            best_prefix = ""
            for token in node.children:
                if key.startswith(token) and len(token) > len(best_prefix):
                    best_prefix = token

            if best_prefix:
                path.append((node, best_prefix))
                node = node.children[best_prefix]
                key = key[len(best_prefix) :]
            else:
                return

        if node.value is not None:
            _free_ray_refs(node.value)
            node.value = None
            self.touch()

        for parent, token in reversed(path):
            child_node = parent.children[token]
            if child_node.value is None and not child_node.children:
                del parent.children[token]
            else:
                break


@ray.remote(num_cpus=0)
class RolloutTraceStore:
    """Actor for managing trace stores (Tries) across different rollout
    sessions."""

    def __init__(self):
        """Initialize the rollout trace store actor."""
        self.sessions: Dict[str, Trie] = {}
        self.objects: Dict[str, ray.ObjectRef] = {}

    def get_or_create(self, session_id: str) -> Trie:
        """Get the Trie for a session, or create one if it doesn't exist.

        Args:
            session_id (str): The session identifier.

        Returns:
            Trie: The Trie instance associated with the session.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = Trie()
        return self.sessions[session_id]

    def get_state(self, session_id: str) -> dict | None:
        """Get lifecycle metadata for a session.

        Args:
            session_id (str): The session identifier.

        Returns:
            dict | None: A snapshot of session metadata, or None when the
                session does not exist.
        """
        trie = self.sessions.get(session_id)
        if trie is None:
            return None
        return {
            "session_id": session_id,
            "state": trie.state.value,
            "updated_at": trie.updated_at,
            "has_object_ref": trie.expert_key in self.objects if trie.expert_key is not None else False,
        }

    def _set_state(
        self,
        session_id: str,
        next_state: TraceState,
    ) -> TraceState:
        """Set a session state and trigger release when needed."""
        trie = self.sessions.get(session_id)
        if trie is None:
            raise KeyError(f"Trace session {session_id!r} does not exist.")

        trie.state = next_state
        trie.touch()
        self._maybe_release(session_id)

        released = session_id not in self.sessions
        return TraceState.RELEASED if released else next_state

    def _maybe_release(self, session_id: str) -> None:
        """Physically release a session once it reaches ToBeReleased."""
        trie = self.sessions.get(session_id)
        if trie is None or trie.state != TraceState.TO_BE_RELEASED:
            return
        self._release_session(session_id, trie)

    def _release_session(self, session_id: str, trie: Trie) -> None:
        """Release trie data and routed expert refs for one session."""
        if trie.expert_key is not None:
            obj_ref = self.objects.pop(trie.expert_key, None)
            if obj_ref is not None:
                _free_ray_refs(obj_ref)
        trie.release()
        self.sessions.pop(session_id, None)

    def mark_rollout_status(
        self,
        session_id: str,
        status: Status,
        *,
        enable_partial_rollout: bool = False,
    ) -> str:
        """Apply a rollout-side status event to one trace session."""
        release_like = status in _ROLLOUT_RELEASE_STATUSES or (
            status == Status.ABORTED and not enable_partial_rollout
        )

        trie = self.sessions.get(session_id)
        if trie is None:
            if release_like:
                return TraceState.RELEASED.value
            raise KeyError(f"Trace session {session_id!r} does not exist.")
        if trie.state != TraceState.ROLLOUT_RUNNING:
            raise RuntimeError(
                f"Cannot handle mark_rollout_status for trace session {session_id!r} "
                f"in state {trie.state.value}."
            )

        if release_like:
            return self._set_state(session_id, TraceState.TO_BE_RELEASED).value
        if status == Status.COMPLETED:
            return self._set_state(session_id, TraceState.ROLLOUT_FINISHED).value
        if status in _ROLLOUT_KEEP_RUNNING_STATUSES:
            trie.touch()
            return TraceState.ROLLOUT_RUNNING.value
        raise AssertionError(f"Unhandled rollout status: {status!r}")

    def mark_commit_failed(self, session_id: str) -> str:
        """Release a rollout session whose response commit failed."""
        trie = self.sessions.get(session_id)
        if trie is None:
            return TraceState.RELEASED.value
        if trie.state != TraceState.ROLLOUT_RUNNING:
            raise RuntimeError(
                f"Cannot handle mark_commit_failed for trace session {session_id!r} "
                f"in state {trie.state.value}."
            )
        return self._set_state(session_id, TraceState.TO_BE_RELEASED).value

    def mark_rollout_discarded(self, session_id: str) -> str:
        """Release a rollout session that external scheduling has discarded."""
        trie = self.sessions.get(session_id)
        if trie is None:
            return TraceState.RELEASED.value
        if trie.state not in (TraceState.ROLLOUT_RUNNING, TraceState.ROLLOUT_FINISHED):
            raise RuntimeError(
                f"Cannot handle mark_rollout_discarded for trace session {session_id!r} "
                f"in state {trie.state.value}."
            )
        return self._set_state(session_id, TraceState.TO_BE_RELEASED).value

    def keys(self, session_id: str) -> List[str]:
        """Get all keys (i.e., strings) stored in a session's Trie.

        Args:
            session_id (str): The session identifier.

        Returns:
            List[str]: A list of all keys in the session's Trie.
        """
        trie = self.get_or_create(session_id)
        return trie.keys()

    def insert(
        self,
        session_id: str,
        key: str,
        value: TokenizedSegment,
        routed_experts: ray.ObjectRef | None = None,
    ):
        """Insert a (key, value) pair into a session's Trie.

        Args:
            session_id (str): The session identifier.
            key (str): The key string.
            value (TokenizedSegment): The trace segment to store.
            routed_experts (ray.ObjectRef | None): Optional routed experts
                object for this session.
        """
        trie = self.get_or_create(session_id)
        if routed_experts is not None:
            expert_key = make_expert_key(session_id)
            self.objects[expert_key] = routed_experts
            value.expert_key = expert_key
            trie.expert_key = expert_key
        trie.insert(key, value)

    def search(self, session_id: str, text: str, filter_none: bool = False):
        """Search the longest prefix in a session's Trie.

        Args:
            session_id (str): The session identifier.
            text (str): The input text to search.
            filter_none (bool): Whether to filter out nodes with no value.

        Returns:
            Tuple[str, List["TreeNode"]]: The matched prefix and matched nodes.
        """
        trie = self.get_or_create(session_id)
        return trie.search(text, filter_none)

    def export_training_trace(self, session_id: str, prompt_text: str) -> dict:
        """Export the stored training trace given a complete prompt text.

        Args:
            session_id (str): The session identifier.
            prompt_text (str): The complete assembled prompt string to look up.

        Returns:
            dict: The trace dictionary containing `input_ids`, `labels`, `logprobs`,
                and the session-level `routed_experts` object key.

        Raises:
            KeyError: If the session does not exist.
            RuntimeError: If the session is not ready for training export.
            ValueError: If the prompt_text does not completely match the trace keys in the session.
        """
        trie = self.sessions.get(session_id)
        if trie is None:
            raise KeyError(f"Trace session {session_id!r} does not exist.")
        if trie.state != TraceState.ROLLOUT_FINISHED:
            raise RuntimeError(
                f"Cannot export training trace for session {session_id!r} in state {trie.state.value}."
            )

        key, nodes = trie.search(prompt_text, filter_none=True)
        if prompt_text != key:
            self._set_state(session_id, TraceState.TO_BE_RELEASED)
            raise ValueError(
                f"Prompt text '{prompt_text}' does not match any trace key '{key}' in session '{session_id}'."
            )
        trace = {"input_ids": [], "labels": [], "logprobs": [], "routed_experts": trie.expert_key}
        for node in nodes:
            node_val: TokenizedSegment = node.value
            trace["input_ids"].extend(node_val.token_ids)
            trace["labels"].extend(node_val.labels)
            trace["logprobs"].extend(node_val.logprobs)
        self._set_state(session_id, TraceState.TRAIN_RUNNING)
        return trace

    def mark_train_finished(self, session_id: str) -> str:
        """Release a session after trainer consumers have finished using it."""
        trie = self.sessions.get(session_id)
        if trie is None:
            return TraceState.RELEASED.value
        if trie.state != TraceState.TRAIN_RUNNING:
            raise RuntimeError(
                f"Cannot handle mark_train_finished for trace session {session_id!r} "
                f"in state {trie.state.value}."
            )
        self._set_state(session_id, TraceState.TRAIN_FINISHED)
        return self._set_state(session_id, TraceState.TO_BE_RELEASED).value

    def mark_train_abandoned(self, session_id: str) -> str:
        """Release a training session that trainer will no longer consume."""
        trie = self.sessions.get(session_id)
        if trie is None:
            return TraceState.RELEASED.value
        if trie.state != TraceState.TRAIN_RUNNING:
            raise RuntimeError(
                f"Cannot handle mark_train_abandoned for trace session {session_id!r} "
                f"in state {trie.state.value}."
            )
        return self._set_state(session_id, TraceState.TO_BE_RELEASED).value

    def get_objects(self, keys: list[str]) -> list[ray.ObjectRef]:
        """Fetch ray.ObjectRef elements by their keys.

        Args:
            keys (list[str]): The list of object keys to retrieve.

        Returns:
            list[ray.ObjectRef]: The mapped ray.ObjectRefs.
        """
        object_refs: list[ray.ObjectRef] = []
        for key in keys:
            if not isinstance(key, str) or not key:
                raise KeyError(f"Invalid trace object key: {key!r}")
            if key not in self.objects:
                raise KeyError(f"Trace object key {key!r} does not exist.")
            object_refs.append(self.objects[key])
        return object_refs


def get_store():
    """Process-local cached handle to the singleton store actor.

    Fast path: ``ray.get_actor`` if another caller has already created it.
    Slow path: create the actor under the reserved name; race with concurrent
    creators is handled by catching the "name already taken" ValueError and
    re-looking-up.

    Returns:
        ActorHandle: Handle to the ``RolloutTraceStore`` actor.
    """
    global _handle_cache
    if _handle_cache is not None:
        return _handle_cache

    try:
        _handle_cache = ray.get_actor(_STORE_NAME)
        return _handle_cache
    except ValueError:
        pass

    import time as _time

    for attempt in range(10):
        try:
            _handle_cache = RolloutTraceStore.options(name=_STORE_NAME).remote()
            return _handle_cache
        except ValueError as exc:
            try:
                _handle_cache = ray.get_actor(_STORE_NAME)
                return _handle_cache
            except ValueError:
                get_logger().debug(f"RolloutTraceStore bootstrap retry {attempt}: {exc}")
                _time.sleep(0.2 * (attempt + 1))
                continue

    raise RuntimeError(f"RolloutTraceStore: failed to acquire named actor {_STORE_NAME!r} after retries")


if __name__ == "__main__":
    print("=== 评估使用 Trie 加速 tokenize.py 避免多轮对话重复 tokenization ===")

    # 1. 初始化
    trie = Trie()

    # 2. 模拟增量 Tokenize 输出
    def mock_segment_tokenize(segment_text: str, is_assistant_response: bool):
        tok_len = max(1, len(segment_text) // 4)
        if is_assistant_response:
            return {
                "segment_text": segment_text,
                "token_ids": [1000] * tok_len,
                "labels": [1000] * tok_len,  # 模型真实生成内容，参与 Loss 计算
                "logprobs": [-0.1] * tok_len,  # 拥有真实的概率分布
            }
        else:
            return {
                "segment_text": segment_text,
                "token_ids": [1000] * tok_len,
                "labels": [-100] * tok_len,  # 模板/用户文本被掩蔽 (Mask)
                "logprobs": [0.0] * tok_len,
            }

    # 3. 定义两轮对话的四个增量片段，明确分离“非模型生成段”与“模型生成段”
    non_asst_str1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
    asst_str1 = "Hi there!<|im_end|>"
    non_asst_str2 = "\n<|im_start|>user\nIntroduce yourself.<|im_end|>\n<|im_start|>assistant\n"
    asst_str2 = "I am an AI assistant created to help you.<|im_end|>"

    # 4. 模拟运行流：我们在每收到一个新的增量片段时，将“至今为止的整个 Prefix String”作为 key，
    #    且把“最新计算好的增量 Token 状态”作为 value 存入 Trie。

    print("\n[Step 1] 处理第一轮 non_assistant 提示...")
    current_prefix = non_asst_str1
    val1 = mock_segment_tokenize(non_asst_str1, is_assistant_response=False)
    trie.insert(current_prefix, val1)

    print("[Step 2] 模型生成了第一轮回复，将其追加...")
    current_prefix += asst_str1
    val2 = mock_segment_tokenize(asst_str1, is_assistant_response=True)
    trie.insert(current_prefix, val2)

    print("[Step 3] 追加第二轮 non_assistant 提示...")
    current_prefix += non_asst_str2
    val3 = mock_segment_tokenize(non_asst_str2, is_assistant_response=False)
    trie.insert(current_prefix, val3)

    # 5. 模拟一次线上推理 / 强化学习推演的过程
    # 假设我们现在需要对一段更长的请求（第二轮模型开始生成前）的完整 prompt 进行处理：
    turn2_prompt = non_asst_str1 + asst_str1 + non_asst_str2

    print(f"\n[Search] 收到需要 tokenizer 计算的长 Prompt (总长 {len(turn2_prompt)} 字符)")
    prefix, nodes = trie.search(turn2_prompt)

    # 6. 从前缀树中收集沿路径出现的所有 value
    cached_length = 0
    reconstructed_labels = []
    reconstructed_string_parts = []

    for node in nodes:
        if node.value is not None:
            val = node.value
            cached_length += len(val["segment_text"])
            reconstructed_labels.extend(val["labels"])
            reconstructed_string_parts.append(val["segment_text"])
            print(
                f"   -> 命中缓存片段: {repr(val['segment_text'][:30])}..., 长度: {len(val['segment_text'])}, 掩蔽标记({val['labels'][0]}): {len(val['labels'])} tokens"
            )

    reconstructed_string = "".join(reconstructed_string_parts)
    print(f"\n   -> 测试字符串重建: {reconstructed_string == turn2_prompt[:cached_length]}")

    print("\n[Result] 统计验证：")
    print(f"  -> Cache 命中的总字符长度: {cached_length} / {len(turn2_prompt)}")

    if cached_length == len(turn2_prompt):
        print("  -> 🥳 完美！整个 Prompt 都命中了 Trie 缓存内容，完全不需要调用 tokenizer.encode 进行重复处理！")
    else:
        unmatched_text = turn2_prompt[cached_length:]
        print(f"  -> 剩下实际需要交给 tokenizer 的未命中部分: {unmatched_text!r}")
    trie.release()
