"""GDP PDF agent loop configuration over the generic multi-turn tool loop."""

from __future__ import annotations

from collections.abc import Callable

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.agent_loop import MultiTurnToolAgentLoopConfig, ToolExecutor

from .tools import GdpDocTools


class GdpToolAgentLoopConfig(MultiTurnToolAgentLoopConfig):
    """Configuration for the GDP PDF tool agent loop.

    Args:
        max_tool_response_chars (int): Maximum characters returned by a tool.
        page_image_zoom (float): Zoom used when rendering a PDF page image.
    """

    max_tool_response_chars: int = 4000
    page_image_zoom: float = 2.0
    enable_batch_judge: bool = True

    def build_tool_executor_factory(self) -> Callable[[RolloutState], ToolExecutor]:
        max_chars = self.max_tool_response_chars
        zoom = self.page_image_zoom

        def factory(state: RolloutState) -> ToolExecutor:
            return GdpDocTools(state.extra_fields.get("pdf_path", ""), max_chars, image_zoom=zoom)

        return factory
