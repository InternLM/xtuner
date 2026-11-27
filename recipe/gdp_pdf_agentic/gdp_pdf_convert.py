"""Convert GDP.pdf (pdf+prompt+rubric) into VLMJsonlDataset jsonl for multimodal rubric RL.

Usage:
  python scripts/gdp_pdf_convert.py [--rubric-version v2] [--max-pages 32] [--workers 8]

Outputs under data/gdp_pdf/ (default --rubric-version v2, never overwrites v1 files):
  gdp_rubric_v2.jsonl / gdp_rubric_smoke_v2.jsonl
  gdp_v2_config.json  / gdp_smoke_v2_config.json
"""

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed


warnings.filterwarnings("ignore")

D = "/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/glm53-multimodal-agentic-rl/data/gdp_pdf"
MEDIA_ROOT = "/mnt/shared-storage-user/llmrazor-share/yushan/glm53-multimodal-agentic-rl/data/gdp_pdf_rendered"
LONG_SIDE = 1120  # px, renders to ~1000-1600 vision tokens/page on GLM-5.3 processor

SEVERITY_WEIGHT_V1 = {
    "Certain dealbreaker": 1.0,
    "Possible dealbreaker": 0.5,
    "Unlikely dealbreaker": 0.25,
}
# 开源 / PRM 口径（v2）：Primary Intent 按 1.0 / 0.6 / 0.3 进分子与分母；
# Dodged Bullet 按 dodged_bullet 类型（默认权重 0.6）只扣分、不进分母。
PI_SEVERITY_WEIGHT_V2 = {
    "Certain dealbreaker": 1.0,
    "Possible dealbreaker": 0.6,
    "Unlikely dealbreaker": 0.3,
}
DEFAULT_DB_WEIGHT_V2 = 0.6


def render_pdf(args):
    pdf_rel, max_pages, re_render = args
    import pymupdf
    from PIL import Image

    pdf_path = os.path.join(D, pdf_rel)
    stem = os.path.splitext(os.path.basename(pdf_rel))[0]
    out_dir = MEDIA_ROOT
    os.makedirs(out_dir, exist_ok=True)
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        return pdf_rel, None, f"open failed: {e}"
    total_pages = doc.page_count
    n_pages = min(total_pages, max_pages)
    images = []
    try:
        for i in range(n_pages):
            fname = f"{stem}_p{i:03d}.png"
            fpath = os.path.join(out_dir, fname)
            if not re_render and os.path.exists(fpath):
                with Image.open(fpath) as im:
                    w, h = im.size
                images.append({"url": fname, "image_wh": [w, h]})
                continue
            page = doc.load_page(i)
            rect = page.rect
            zoom = LONG_SIDE / max(rect.width, rect.height)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            pix.save(fpath)
            images.append({"url": fname, "image_wh": [pix.width, pix.height]})
    except Exception as e:
        return pdf_rel, None, f"render failed at page {i}: {e}"
    finally:
        doc.close()
    return pdf_rel, {"n_pages_total": total_pages, "images": images}, None


def build_entry(row, render_info, rubric_version="v2", db_weight=DEFAULT_DB_WEIGHT_V2):
    rubrics = []
    for i in range(1, 31):
        crit = row.get(f"rubric - {i}. criterion")
        if not isinstance(crit, str) or not crit.strip():
            continue
        raw_sev = row.get(f"rubric - {i}. criterion_severity")
        sev = raw_sev.strip() if isinstance(raw_sev, str) and raw_sev.strip() else None
        raw_type = row.get(f"rubric - {i}. criterion_type")
        orig_type = raw_type.strip() if isinstance(raw_type, str) and raw_type.strip() else "criterion"

        if rubric_version == "v1":
            r_type = "answer_coverage"
            weight = SEVERITY_WEIGHT_V1.get(sev, 1.0)
        else:
            if orig_type == "Dodged Bullet":
                r_type = "dodged_bullet"
                weight = float(db_weight)
            else:
                r_type = "answer_coverage"
                weight = PI_SEVERITY_WEIGHT_V2.get(sev, 1.0)

        rubrics.append(
            {
                "local_id": f"c{i}",
                "type": r_type,
                "orig_type": orig_type,
                "severity": sev,
                "criterion": crit.strip(),
                "weight": weight,
            }
        )
    if not rubrics:
        return None
    pos_weight = sum(r["weight"] for r in rubrics if r["type"] != "dodged_bullet")
    neg_weight = sum(r["weight"] for r in rubrics if r["type"] == "dodged_bullet")
    total_weight = sum(r["weight"] for r in rubrics) if rubric_version == "v1" else pos_weight
    content = [{"type": "image", "image": img} for img in render_info["images"]]
    content.append({"type": "text", "text": row["prompt"]})
    extra_info = {
        "domain": row["domain"],
        "task_id": row["task_id"],
        "n_pages_total": render_info["n_pages_total"],
        "n_pages_rendered": len(render_info["images"]),
        "rubrics": rubrics,
        "total_weight": round(total_weight, 6),
    }
    if rubric_version != "v1":
        extra_info["positive_weight"] = round(pos_weight, 6)
        extra_info["penalty_weight"] = round(neg_weight, 6)
        extra_info["rubric_version"] = rubric_version
    return {
        "data_source": "gdp_pdf",
        "prompt": [{"role": "user", "content": content}],
        "ability": "document_qa",
        "reward_model": {"ground_truth": "", "style": "rubric"},
        "extra_info": extra_info,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-smoke", action="store_true")
    ap.add_argument("--re-render", action="store_true", help="force re-rendering existing PNGs")
    ap.add_argument("--rubric-version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--dodged-bullet-weight", type=float, default=DEFAULT_DB_WEIGHT_V2)
    ap.add_argument("--output-suffix", type=str, default=None)
    args = ap.parse_args()

    suffix = args.output_suffix if args.output_suffix is not None else ("" if args.rubric_version == "v1" else "_v2")

    import pandas as pd

    df = pd.read_parquet(os.path.join(D, "data.parquet"))
    pdf_rels = list(df["pdf_path"])

    render_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(render_pdf, (p, args.max_pages, args.re_render)): p for p in pdf_rels}
        for fut in as_completed(futs):
            pdf_rel, info, err = fut.result()
            if err:
                print(f"ERROR {pdf_rel}: {err}", file=sys.stderr)
            else:
                render_results[pdf_rel] = info
    print(f"prepared {len(render_results)}/{len(pdf_rels)} pdfs", file=sys.stderr)

    entries = []
    for _, row in df.iterrows():
        info = render_results.get(row["pdf_path"])
        if not info:
            continue
        entry = build_entry(row, info, rubric_version=args.rubric_version, db_weight=args.dodged_bullet_weight)
        if entry:
            entries.append(entry)

    entries.sort(key=lambda e: e["extra_info"]["n_pages_total"])
    smoke = [e for e in entries if e["extra_info"]["n_pages_total"] <= 8][:8]

    anno = os.path.join(D, f"gdp_rubric{suffix}.jsonl")
    anno_smoke = os.path.join(D, f"gdp_rubric_smoke{suffix}.jsonl")
    cfg_path = os.path.join(D, f"gdp{suffix}_config.json")
    cfg_smoke_path = os.path.join(D, f"gdp_smoke{suffix}_config.json")

    if not args.only_smoke:
        with open(anno, "w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n")
        with open(cfg_path, "w") as f:
            json.dump({"gdp_pdf": {"annotation": [anno], "media_root": MEDIA_ROOT, "sample_ratio": 1.0}}, f, indent=2)

    with open(anno_smoke, "w") as f:
        for e in smoke:
            f.write(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n")
    smoke_key = "gdp_pdf_smoke" if args.rubric_version == "v1" else "gdp_pdf"
    with open(cfg_smoke_path, "w") as f:
        json.dump(
            {smoke_key: {"annotation": [anno_smoke], "media_root": MEDIA_ROOT, "sample_ratio": 1.0}}, f, indent=2
        )

    n_img = sum(e["extra_info"]["n_pages_rendered"] for e in entries)
    print(
        f"version={args.rubric_version} entries={len(entries)} smoke={len(smoke)} "
        f"total_images={n_img} -> {anno}, {anno_smoke}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
