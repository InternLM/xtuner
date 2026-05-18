from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.judger.native import Judger


CALCULATOR_TOOL_NAME = "mcp__calculator__calculator"
CALCULATOR_PROMPT = """You are NOT allowed to do arithmetic yourself.

You MUST use the calculator tool to compute the result.
The calculator tool is named mcp__calculator__calculator.
Your first assistant response MUST be exactly one structured tool call and nothing else:
<tool_call>
<function=mcp__calculator__calculator>
<parameter=expression>
23 + 19
</parameter>
</function>
</tool_call>

Question:
What is 23 + 19?

Return only the final answer."""

CALCULATOR_SYSTEM_PROMPT = """You are testing an agent loop tool-calling path.
The only successful behavior is:
1. Call the mcp__calculator__calculator tool with {"expression": "23 + 19"}.
2. Read the tool result.
3. Return only the final answer as plain text: 42.

Do not solve arithmetic directly. Do not describe a tool call in prose.
Do not generate a title. Do not use boxed answer formatting.
For Qwen-style tool calls, use this exact XML form:
<tool_call>
<function=mcp__calculator__calculator>
<parameter=expression>
23 + 19
</parameter>
</function>
</tool_call>"""


class CalculatorJudger(Judger):
    async def judge(self, rollout_state: RolloutState) -> RolloutState:
        stdout = rollout_state.extra_fields.get("claudecode_cli_stdout") or ""
        answer = ""
        if stdout:
            try:
                answer = normalize_answer(json.loads(stdout).get("result"))
            except Exception:
                answer = normalize_answer(stdout)
        rollout_state.reward = {
            "score": 1.0 if answer == "42" else 0.0,
            "answer": answer,
        }
        return rollout_state


def normalize_answer(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().strip("`").strip()
    boxed = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()
    final_answer = re.search(r"final answer\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if final_answer:
        text = final_answer.group(1).strip()
    return text.strip().strip("`").strip()


def write_calculator_mcp_server(work_dir: Path) -> tuple[Path, Path]:
    mcp_server_path = work_dir / "calculator_mcp_server.py"
    mcp_config_path = work_dir / "calculator_mcp_config.json"
    mcp_server_path.write_text(
        """
from __future__ import annotations

import ast
import operator

from fastmcp import FastMCP


mcp = FastMCP("calculator")

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Only simple arithmetic expressions are supported.")


@mcp.tool(name="calculator", description="Evaluate a simple arithmetic expression")
def calculator(expression: str) -> str:
    value = _eval(ast.parse(expression, mode="eval"))
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


if __name__ == "__main__":
    mcp.run()
""".lstrip(),
        encoding="utf-8",
    )
    mcp_config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "calculator": {
                        "command": sys.executable,
                        "args": [str(mcp_server_path)],
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return mcp_server_path, mcp_config_path
