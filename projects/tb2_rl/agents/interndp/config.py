"""Agent config for interndp (tb2-rl).  Paths read from env at daemon-start time."""

import os

from lagent.agents.fc_agent import get_tool_prompt

tool_template = """# Tools

You have access to the following functions:

<tools>
{tools}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""

workspace = os.environ.get("TASK_WORKSPACE", "/app")
skills_root = f"{workspace}/skills"

model = dict(
    type="lagent.llms.model.AsyncAPIClient",
    model=dict(
        model=os.environ.get(
            "RL_LLM_MODEL",
            "",
        ),
        base_url=os.environ.get(
            "RL_LLM_BASE_URL",
            "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1",
        ),
        api_key=os.environ.get("RL_LLM_API_KEY", "sk-admin"),
    ),
    sample_params=dict(temperature=0.7, top_p=1.0, top_k=50),
    timeout=900,
    max_retry=1,
    sleep_interval=5,
    extra_body=dict(spaces_between_special_tokens=False),
)

base_actions = [dict(type="lagent.actions.tmux_action.TerminalExecute")]

policy_agent = dict(
    type="lagent.agents.AsyncAgent",
    llm=model,
    template=get_tool_prompt(base_actions, template=tool_template),
    hooks=[dict(type="lagent.hooks.logger.MessageLogger")],
)

env_agent = dict(
    type="lagent.agents.env_agent.RLEnvAgent",
    actions=base_actions,
    max_turn=100,
    max_tool_response_length=8192,
    tool_response_truncate_side="middle",
    enable_no_thinking_penalty=False,
)

agent_config = dict(
    type="lagent.agents.fc_agent.FunctionCallAgent",
    policy_agent=policy_agent,
    env_agent=env_agent,
    finish_condition='lagent.agents.env_agent.finish_condition_func',
    initialize_input=False,
)


if __name__ == "__main__":
    import asyncio

    from lagent.utils import create_object

    async def main():
        agent = create_object(agent_config)
        res = await agent("list the files in the current directory and tell me which is the largest")
        print(res)

    asyncio.run(main())
