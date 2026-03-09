import re

import aiohttp
from transformers.utils import get_json_schema

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse


class SandboxTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```py(.*?)```", re.DOTALL)

    async def code_interpreter(self, code: str) -> str:
        """Execute the code in the sandbox.

        Args:
            code: The code to be executed.

        Returns:
            str: The output of the code execution.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.get("sandbox_fusion_url"),
                json={"code": code},
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                stdout, stderr = result["run_result"]["stdout"], result["run_result"]["stderr"]
                return stdout + stderr

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.code_interpreter)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[str, float, dict]:
        code = parameters["code"]
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        result = await self.code_interpreter(code)
        return ToolResponse(text=result), 0.0, {}
