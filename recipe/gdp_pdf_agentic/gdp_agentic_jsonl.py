#!/usr/bin/env python
"""把 GDP rubric jsonl 转成 agentic 版（带工具 schema、工具使用说明、pdf_path）。

用法: python scripts/gdp_agentic_jsonl.py [输入jsonl] [输出jsonl] [--profile glm53|qwen35_vl] [--max-inline-pages N] [--write-config]
默认输入 data/gdp_pdf/gdp_rubric_smoke_v2.jsonl，输出同目录对应 agentic jsonl（不覆盖 v1 历史文件）。
"""

import json
from pathlib import Path


PROJ = Path(__file__).resolve().parents[1]
PDF_DIR = PROJ / "data/gdp_pdf/pdfs"
DEFAULT_MEDIA_ROOT = "/mnt/shared-storage-user/llmrazor-share/yushan/glm53-multimodal-agentic-rl/data/gdp_pdf_rendered"

GDP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_doc_meta",
            "description": "Get document metadata, e.g. the total page count.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_text",
            "description": "Extract the text layer of a specific page (1-based) of the document.",
            "parameters": {
                "type": "object",
                "properties": {"page": {"type": "integer", "description": "1-based page number"}},
                "required": ["page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_image",
            "description": "Render a specific page (1-based) of the document as an image and attach it to the tool result.",
            "parameters": {
                "type": "object",
                "properties": {"page": {"type": "integer", "description": "1-based page number"}},
                "required": ["page"],
            },
        },
    },
]

INSTRUCTION_SUFFIX = (
    "\n\nYou may inspect the document further with tools before answering. "
    "The attached image(s) show only part of the document; it has {n_pages} page(s) in total. "
    "To call a tool, output exactly one call in this format and then stop:\n"
    "<tool_call>TOOL_NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value></tool_call>\n"
    "Available tools: get_doc_meta() returns the page count; "
    "get_page_text(page) returns the text layer of one page (1-based); "
    "get_page_image(page) renders one page as an image so you can read figures and layout. "
    "After each tool result, continue: call another tool if needed, "
    "or give your final answer directly without any tool call."
)

# qwen35_vl 与 loop.py parse_qwen_tool_call（JSON 格式）对齐
INSTRUCTION_SUFFIX_QWEN = (
    "\n\nYou may inspect the document further with tools before answering. "
    "The attached image(s) show only part of the document; it has {n_pages} page(s) in total. "
    "To call a tool, output exactly one call in this format and then stop:\n"
    '<tool_call>{{"name": "TOOL_NAME", "arguments": {{"KEY": VALUE}}}}</tool_call>\n'
    "Available tools: get_doc_meta() returns the page count; "
    "get_page_text(page) returns the text layer of one page (1-based); "
    "get_page_image(page) renders one page as an image so you can read figures and layout. "
    "After each tool result, continue: call another tool if needed, "
    "or give your final answer directly without any tool call."
)


def pdf_path_for(item: dict) -> str:
    """从首个图片 url 推导 pdf 路径（渲染命名：<pdf_stem>_pNNN.png）。"""
    for msg in item["prompt"]:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                stem = part["image"]["url"].rsplit("_p", 1)[0]
                path = PDF_DIR / f"{stem}.pdf"
                if not path.exists():
                    raise FileNotFoundError(f"pdf not found: {path}")
                return str(path)
    raise ValueError("no image part found in prompt")


def cap_inline_images(item: dict, max_inline_pages: int) -> int:
    """prompt 内联页图只保留前 max_inline_pages 张（按页序排列），其余交给工具按需取。

    返回实际保留的图数。长文档（中位 26.5 页渲染图 ≈ 6.6 万 token）全内联会远超
    max_prompt_length，且 Qwen RL tokenize 对超长 prompt 只 WARNING 截断（不报错），
    必须在内联侧控量。
    """
    kept = 0
    for msg in item["prompt"]:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                if kept < max_inline_pages:
                    kept += 1
                    new_content.append(part)
                continue
            new_content.append(part)
        msg["content"] = new_content
    return kept


def write_manifest(dst: Path, media_root: str = DEFAULT_MEDIA_ROOT) -> Path:
    cfg_path = dst.with_name(f"{dst.stem}_config.json")
    manifest = {
        "gdp_pdf": {
            "annotation": [str(dst.resolve())],
            "media_root": media_root,
            "sample_ratio": 1.0,
        }
    }
    cfg_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote manifest -> {cfg_path}")
    return cfg_path


def convert(
    src: Path,
    dst: Path,
    profile: str = "glm53",
    max_inline_pages: int | None = None,
    write_config: bool = False,
    media_root: str = DEFAULT_MEDIA_ROOT,
) -> None:
    suffix = INSTRUCTION_SUFFIX_QWEN if profile == "qwen35_vl" else INSTRUCTION_SUFFIX
    n = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            item = json.loads(line)
            extra = item.setdefault("extra_info", {})
            extra["pdf_path"] = pdf_path_for(item)
            if max_inline_pages is not None:
                extra["n_pages_inlined"] = cap_inline_images(item, max_inline_pages)
            item["tools"] = GDP_TOOLS
            for msg in item["prompt"]:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] += suffix.format(n_pages=extra.get("n_pages_total", "?"))
            fout.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            n += 1
    print(f"wrote {n} samples ({profile}) -> {dst}")
    if write_config:
        write_manifest(dst, media_root=media_root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("src", nargs="?", default=str(PROJ / "data/gdp_pdf/gdp_rubric_smoke_v2.jsonl"))
    parser.add_argument("dst", nargs="?", default=None)
    parser.add_argument("--profile", choices=["glm53", "qwen35_vl"], default="glm53")
    parser.add_argument(
        "--max-inline-pages",
        type=int,
        default=None,
        help="prompt 内联页图上限（其余页交给工具按需取）；默认 None 保留全部内联",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="同步写出 <dst_stem>_config.json 数据集清单",
    )
    parser.add_argument("--media-root", type=str, default=DEFAULT_MEDIA_ROOT)
    args = parser.parse_args()
    src = Path(args.src)
    if args.dst:
        dst = Path(args.dst)
    elif args.profile == "qwen35_vl":
        dst = src.with_name(src.stem.replace("rubric", "agentic_qwen") + ".jsonl")
    else:
        dst = src.with_name(src.stem.replace("rubric", "agentic") + ".jsonl")
    convert(src, dst, args.profile, args.max_inline_pages, write_config=args.write_config, media_root=args.media_root)
