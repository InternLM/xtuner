"""GDP 文档工具：基于 pymupdf 的页面文本提取（M1）与页面截图（M2）。"""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass


try:
    import pymupdf
except ImportError:  # 旧版包名
    import fitz as pymupdf  # type: ignore[no-redef]


@dataclass
class ToolOutput:
    """工具结果：文本必带；图片工具额外带 data URI（供 rollout image_data 与判分透传）。"""

    text: str
    image_data: str | None = None


class GdpDocTools:
    """按样本加载 PDF，提供 get_doc_meta / get_page_text / get_page_image。"""

    def __init__(self, pdf_path: str, max_chars: int = 4000, image_zoom: float = 2.0):
        if not pdf_path:
            raise ValueError("GdpDocTools requires pdf_path")
        self.pdf_path = pdf_path
        self.max_chars = max_chars
        self.image_zoom = image_zoom
        self._lock = threading.Lock()
        self._doc = pymupdf.open(pdf_path)

    def close(self) -> None:
        self._doc.close()

    def get_doc_meta(self) -> str:
        return f'{{"n_pages": {self._doc.page_count}}}'

    def get_page_text(self, page: int) -> str:
        with self._lock:
            if not 1 <= page <= self._doc.page_count:
                return f"Error: page {page} out of range [1, {self._doc.page_count}]"
            text = self._doc[page - 1].get_text("text")
        text = text.strip()
        if not text:
            return f"(page {page} has no extractable text layer)"
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + f"... [truncated at {self.max_chars} chars]"
        return text

    def get_page_image(self, page: int) -> ToolOutput:
        """渲染指定页为 PNG，返回 data URI；文本部分给 agent 一个简短说明。"""
        with self._lock:
            if not 1 <= page <= self._doc.page_count:
                return ToolOutput(f"Error: page {page} out of range [1, {self._doc.page_count}]")
            pixmap = self._doc[page - 1].get_pixmap(
                matrix=pymupdf.Matrix(self.image_zoom, self.image_zoom), alpha=False
            )
            png_bytes = pixmap.tobytes("png")
            width, height = pixmap.width, pixmap.height
        data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        return ToolOutput(f"(page {page} rendered as {width}x{height} image, attached below)", data_uri)

    def call(self, name: str, arguments: dict) -> ToolOutput:
        if name == "get_doc_meta":
            return ToolOutput(self.get_doc_meta())
        if name == "get_page_text":
            try:
                page = int(arguments.get("page"))
            except (TypeError, ValueError):
                return ToolOutput(f"Error: invalid page argument {arguments.get('page')!r}")
            return ToolOutput(self.get_page_text(page))
        if name == "get_page_image":
            try:
                page = int(arguments.get("page"))
            except (TypeError, ValueError):
                return ToolOutput(f"Error: invalid page argument {arguments.get('page')!r}")
            return self.get_page_image(page)
        return ToolOutput(f"Error: unknown tool {name!r}")
