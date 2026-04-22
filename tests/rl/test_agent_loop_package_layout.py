from xtuner.v1.rl import agent_loop, agent_loop_manager


def test_agent_loop_package_only_exports_loop_objects():
    # 这里明确包边界：agent_loop 只放单条 rollout loop，不再兼容导出 manager 对象。
    assert hasattr(agent_loop, "AgentLoop")
    assert hasattr(agent_loop, "SingleTurnAgentLoopConfig")
    assert not hasattr(agent_loop, "AgentLoopManagerConfig")
    assert not hasattr(agent_loop, "SamplerConfig")
    assert not hasattr(agent_loop, "ProduceBatchStatus")


def test_agent_loop_manager_package_exports_manager_objects():
    # manager 包统一承载批量调度、采样和 producer 相关类型。
    assert hasattr(agent_loop_manager, "AgentLoopManagerConfig")
    assert hasattr(agent_loop_manager, "SamplerConfig")
    assert hasattr(agent_loop_manager, "ProduceBatchStatus")
    assert hasattr(agent_loop_manager, "ProduceBatchResult")
