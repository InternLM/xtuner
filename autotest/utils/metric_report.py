"""Generate e2e metric comparison plots and GitHub Actions job summaries."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Keep GitHub Step Summary small; full tracker.jsonl stays on cluster baseline paths.
SUMMARY_JSONL_PREVIEW_LINES = 3

DEFAULT_RAW_URL_BASE = "https://raw.githubusercontent.com/llmcitester/xtuner/reports/e2e"


def get_report_dir() -> Path:
    run_id = os.environ.get("GITHUB_RUN_ID", "0")
    report_dir = Path.cwd() / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def report_image_url(case_name: str) -> str:
    run_id = os.environ.get("GITHUB_RUN_ID", "0")
    raw_base = os.environ.get("CI_REPORTS_RAW_URL_BASE", "").rstrip("/") or DEFAULT_RAW_URL_BASE
    device = os.environ.get("DEVICE", "")
    prefix = f"{raw_base}/npu" if device == "npu" else raw_base
    return f"{prefix}/{run_id}/{case_name}_comparison.png"


def plot_comparison(
    case_name: str,
    metric_keys: dict,
    base_metrics: dict,
    cur_metrics: dict,
    output_root: Path,
) -> Path:
    metric_list = list(metric_keys.keys())
    n_plots = len(metric_list)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n_plots:
            metric = metric_list[i]
            x_base = np.arange(len(base_metrics[metric]))
            x_current = np.arange(len(cur_metrics[metric]))
            ax.plot(
                x_base,
                base_metrics[metric],
                "r--",
                label="Base",
                marker="x",
                markersize=4,
            )
            ax.plot(
                x_current,
                cur_metrics[metric],
                "b-",
                label="Current",
                marker="o",
                markersize=4,
            )
            ax.set_title(f"{metric.replace('/', '_')}_comparison")
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)
        else:
            ax.axis("off")

    fig.suptitle(f"{case_name}_metrics_comparison", fontsize=16)
    plt.tight_layout()
    output_path = output_root / f"{case_name}_comparison.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    return output_path


def format_jsonl_preview(path: str, label: str) -> str:
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    total = len(lines)
    preview_lines = SUMMARY_JSONL_PREVIEW_LINES
    if total <= preview_lines * 2:
        body_lines = lines
        omitted = 0
    else:
        body_lines = (
            lines[:preview_lines] + [f"... ({total - preview_lines * 2} lines omitted) ..."] + lines[-preview_lines:]
        )
        omitted = total - preview_lines * 2
    md = f"**{label}** (`{path}`, {total} lines"
    if omitted:
        md += f", preview only, {omitted} lines omitted"
    md += ")\n\n```json\n"
    md += "\n".join(body_lines)
    md += "\n```\n\n"
    return md


def append_case_to_step_summary(case_name: str, base_jsonl: str, cur_jsonl: str) -> None:
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY", "./tmp.md")
    image_url = report_image_url(case_name)
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"## {case_name} 指标比较图\n")
        f.write('<div align="center">\n')
        f.write(f'<img src="{image_url}"\n')
        f.write('  style="max-width: 90%; border: 1px solid #ddd; border-radius: 8px;">\n')
        f.write("</div>\n")
        f.write(f"[在 reports 分支查看大图]({image_url})\n\n")
        f.write('<div align="center">\n')
        f.write(
            f'<details>\n<summary><strong style="text-align: left;">'
            f"📊 用例 {case_name} tracker 预览（基线 / 当前，完整数据见集群 baseline 路径）</strong></summary>\n\n"
        )
        f.write(format_jsonl_preview(base_jsonl, "Baseline"))
        f.write(format_jsonl_preview(cur_jsonl, "Current"))
        f.write("</details>\n\n")


def publish_comparison_report(
    case_name: str,
    metric_keys: dict,
    base_metrics: dict,
    cur_metrics: dict,
    base_jsonl: str,
    cur_jsonl: str,
) -> Path:
    """Write comparison PNG under ``{GITHUB_RUN_ID}/`` and append job
    summary."""
    output_root = get_report_dir()
    plot_path = plot_comparison(case_name, metric_keys, base_metrics, cur_metrics, output_root)
    append_case_to_step_summary(case_name, base_jsonl, cur_jsonl)
    return plot_path
