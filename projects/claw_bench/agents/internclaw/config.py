"""Agent config for internclaw.  Paths read from env at daemon-start time."""

import os

workspace = os.environ.get("TASK_WORKSPACE", "/workspace")
skills_root = f"{workspace}/skills"

model = dict(
    type="lagent.llms.model.AsyncAPIClient",
    model=dict(
        model=os.environ.get(
            "RL_LLM_MODEL",
            "sft_interns2_pre_base03_20260413a_lr2e5_128gpu_hf5646",
        ),
        base_url=os.environ.get(
            "RL_LLM_BASE_URL",
            "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1",
        ),
        api_key=os.environ.get("RL_LLM_API_KEY", "sk-admin"),
    ),
    sample_params=dict(temperature=0.7, top_p=1.0, top_k=50),
    timeout=600,
    max_retry=500,
    sleep_interval=5,
    extra_body=dict(spaces_between_special_tokens=False),
)

base_actions = [
    dict(type="lagent.actions.filesystem.ReadFileAction", workspace=workspace),
    dict(type="lagent.actions.filesystem.WriteFileAction", workspace=workspace),
    dict(type="lagent.actions.filesystem.EditFileAction", workspace=workspace),
    dict(type="lagent.actions.shell.ShellAction", working_dir=workspace),
]

save_memory_action = dict(
    type="lagent.actions.save_memory.AsyncSaveMemoryAction",
    workspace=workspace,
)

CONSOLIDATION_PROMPT = (
    "You are a memory consolidation agent. Review the conversation "
    "and call the save_memory tool to persist important information.\n\n"
    "Extract key facts, decisions, user preferences, and project context. "
    "Merge with existing long-term memory. For history_entry, write a "
    "grep-searchable summary starting with [YYYY-MM-DD HH:MM]."
)

policy_agent = dict(
    type="lagent.agents.internclaw_agent.AsyncPolicyAgent",
    llm=model,
    aggregator=dict(
        type="lagent.agents.aggregator.context.InternClawContextBuilder",
        workspace=workspace,
    ),
    hooks=[dict(type="lagent.hooks.logger.MessageLogger")],
)

env_agent = dict(
    type="lagent.agents.internclaw_agent.AsyncEnvAgent",
    actions=base_actions + [save_memory_action],
    skills=dict(
        type="lagent.skills.skills.SkillsLoader",
        workspace=skills_root,
    ),
    long_term_memory=dict(
        type="lagent.memory.openclaw_provider.OpenClawMemoryProvider",
        workspace=workspace,
    ),
)

compact_agent = dict(
    type="lagent.agents.compact_agent.AsyncCompactAgent",
    name="compact",
    llm=model,
    max_context_tokens=65536,
    threshold_ratio=0.5,
)

consolidate_agent = dict(
    type="lagent.agents.internclaw_agent.InternClawAgent",
    policy_agent=dict(
        type="lagent.agents.internclaw_agent.AsyncPolicyAgent",
        name="consolidate_policy",
        llm=model,
        template=CONSOLIDATION_PROMPT,
        hooks=[dict(type="lagent.hooks.logger.MessageLogger")],
        aggregator=dict(type="lagent.agents.aggregator.compact_aggregator.CompactAggregator"),
    ),
    env_agent=dict(
        type="lagent.agents.internclaw_agent.AsyncEnvAgent",
        actions=[save_memory_action],
    ),
    max_turn=1,
    finish_condition=None,
)

agent_config = dict(
    type="lagent.agents.internclaw_agent.InternClawAgent",
    policy_agent=policy_agent,
    env_agent=env_agent,
    compact_agent=compact_agent,
    consolidate_agent=consolidate_agent,
)


if __name__ == '__main__':
    import asyncio

    from lagent.utils import create_object

    async def main():
        agent = create_object(agent_config)
        res = await agent('帮我查看下当前处于哪个目录以及当前目录下有哪些东西？')
        print(res)

    asyncio.run(main())
