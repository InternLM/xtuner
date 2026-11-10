"""Regressions at the new canonical training and opt-in strategy boundaries."""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import ray

from xtuner.v1.data_proto.rl_data import RolloutState, Status, free_rollout_state_refs
from xtuner.v1.rl.agent_loop.sandbox_agent_loop import agent_in_sandbox_loop as sandbox
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import RolloutStatus
from xtuner.v1.rl.agent_loop_manager.agent_turn_pause_policy import AgentTurnPausePolicy
from xtuner.v1.rl.agent_loop_manager.disagg_agent_loop_manager import DisaggAgentLoopManager
from xtuner.v1.rl.agent_loop_manager.disagg_producer import DisaggProduceStrategy
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore, TokenizedSegment
from xtuner.v1.rl.rollout import trace_store


def test_cross_update_trace_with_tool_tokens_matches_main_training_contract(monkeypatch):
    store = RolloutTraceStore.__ray_metadata__.modified_class()
    text = ''
    for word, ids, labels, version, request in [
        ('prompt', [10, 11], [-100, -100], 0, 'r0'),
        ('first', [12], [12], 0, 'r0'),
        ('tool', [13, 14], [-100, -100], 1, 'r1'),
        ('last', [15], [15], 1, 'r1'),
    ]:
        text += word
        store.insert('42', text, TokenizedSegment(
            text=word, token_ids=ids, labels=labels,
            logprobs=[-.3 if label != -100 else 0 for label in labels],
            expert_key=np.full((len(ids), 1, 1), version), model_step=version, request_id=request,
        ))

    # Only the Ray boundary and external tokenizer/Runner result are replaced.
    # Trace export, sandbox conversion, canonicalization and validation are real.
    def remote(method):
        async def call(*args, **kwargs):
            return method(*args, **kwargs)
        return SimpleNamespace(remote=call)

    monkeypatch.setattr(sandbox, 'get_store', lambda: SimpleNamespace(
        export_training_trace=remote(store.export_training_trace),
        contains_request_ids=remote(store.contains_request_ids),
    ))
    messages = [{'role': 'user', 'content': 'task'}, {'role': 'assistant', 'content': 'done'}]
    item = SimpleNamespace(
        artifacts={'response_message': {'content': 'done'}, 'messages': [
            {'messages': messages, 'tools': [], 'request_ids': ['r0', 'r1']}]},
        status=RolloutStatus.COMPLETED, reward=1, judgers={}, data_source='fixture',
        infer=SimpleNamespace(agent=None, sandbox_env_id='fixture'), error=None,
    )
    loop = sandbox.AgentInSandboxLoop.__new__(sandbox.AgentInSandboxLoop)
    loop.mode = 'train'
    loop.require_turn_provenance = True
    loop.logger = MagicMock()
    loop.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: text, decode=lambda ids: 'done')
    states = asyncio.run(loop._build_rollout_states(
        RolloutState(message=[], session_id=42, prompt_ids=[10, 11], group_id=1, rollout_id=1), item
    ))
    loop.canonicalize_train_fields(states)
    assert len(states) == 1
    state = states[0]
    assert state.status == Status.COMPLETED
    state.validate_training()
    assert state.response_ids == [12, 13, 14, 15]
    assert state.response_model_steps == [0, 1, 1, 1]
    assert state.labels == [-100, -100, 12, -100, -100, 15]
    assert state.logprobs == [0, 0, -.3, 0, 0, -.3]
    state.apply_response_mask([False, True, True, True])
    state.validate_training()
    assert state.labels == [-100, -100, -100, -100, -100, 15]


def test_custom_strategy_rejected_at_construction_when_turn_resume_is_enabled():
    class CustomStrategy(DisaggProduceStrategy):
        async def produce_batch(self, ctx):
            pass

    strategy = CustomStrategy(lambda *args: True)
    task = SimpleNamespace(weight=1, produce_strategy=strategy)
    with pytest.raises(ValueError, match='does not support complete-turn resume'):
        DisaggAgentLoopManager([task], None, None, pause_policy=AgentTurnPausePolicy(
            freeze_timeout=1, drain_timeout=1))
    assert 'pending_cleanup' not in vars(strategy)


def test_failed_attempt_filter_is_explicit_recipe_policy():
    valid = RolloutState(message=[], status=Status.COMPLETED, extra_fields={'agent_session_id': 'a'})
    failed = RolloutState(message=[], status=Status.FAILED, extra_fields={'agent_session_id': 'b'})
    group = [valid, failed]
    assert sandbox._drop_failed_train_samples(group, 'train') == [valid]
    assert [item.status for item in group] == [Status.COMPLETED, Status.FAILED]
    assert sandbox._drop_failed_train_samples(group, 'eval', 2) is group
    sandbox._drop_failed_train_samples(group, 'train', 2)
    assert all(item.status == Status.FILTERED for item in group)


@pytest.mark.parametrize('statuses', [
    [Status.COMPLETED, Status.FAILED],
    [Status.FAILED, Status.COMPLETED, Status.COMPLETED],
    [Status.FAILED, Status.FAILED],
    [Status.COMPLETED, Status.COMPLETED],
    [],
])
def test_default_attempt_filter_matches_main_and_preserves_group_consumption(statuses):
    from xtuner.v1.rl.agent_loop_manager.produce_utils import get_group_status

    group = [RolloutState(message=[], status=status, extra_fields={'agent_session_id': str(i)})
             for i, status in enumerate(statuses)]
    expected = [sample for sample in group if sample.status != Status.FAILED] or group
    actual = sandbox._drop_failed_train_samples(group, 'train')
    assert actual == expected
    assert sandbox._drop_failed_train_samples(group, 'eval') is group
    if any(status == Status.COMPLETED for status in statuses):
        assert get_group_status(actual) == Status.COMPLETED


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('operation', ['reset', 'discard'])
def test_ordinary_refs_are_freed_but_request_trace_borrowers_are_preserved(monkeypatch, shared, operation):
    ref = ray.ObjectRef.from_random()
    freed = []
    monkeypatch.setattr('xtuner.v1.rl.utils.ray_utils.free_object_refs', lambda refs: freed.extend([refs] if isinstance(refs, ray.ObjectRef) else refs))
    state = RolloutState(message=[], routed_experts=ref,
                        extra_fields={'request_ids': ['r0']} if shared else {})
    if operation == 'reset':
        state.reset_response()
    else:
        free_rollout_state_refs(state)
    assert state.routed_experts is None
    assert freed == ([] if shared else [ref])


@pytest.mark.parametrize('release_key', [None, 'prompt'])
def test_trace_retirement_preserves_only_request_aware_shared_refs(monkeypatch, release_key):
    freed = []
    monkeypatch.setattr(trace_store, '_free_ray_refs', freed.append)
    plain = TokenizedSegment(text='plain', token_ids=[1])
    shared = TokenizedSegment(text='shared', token_ids=[2], request_id='r0', model_step=0)
    for segment in [plain, shared]:
        trie = trace_store.Trie()
        trie.insert('prompt', segment)
        trie.release(release_key)
    assert freed == [plain]
