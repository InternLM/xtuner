import json
import os
import subprocess
import sys


# Colocate RL trainers write `.xtuner_rl_colocate_trainer` (see rl_trainer._META_PATH).
# Keep legacy / disagg names as fallbacks for older or alternate runs.
RL_META_CANDIDATES = (
    ".xtuner_rl_colocate_trainer",
    ".xtuner_rl_disaggregated_trainer",
    ".xtuner_grpo",
)


def update_meta(ori_meta_file, new_meta):
    with open(ori_meta_file, encoding="utf-8") as f:
        meta_info = json.load(f)
        print(meta_info)

    meta_info["exps"][0]["history"][0]["end"] = new_meta["end"]
    meta_info["exps"][0]["exp_dir"] = new_meta["exp_dir"]
    meta_info["exps"][0]["checkpoint_list"] = new_meta["checkpoint_list"]
    meta_info["exps"][0]["cur_step"] = new_meta["end"]

    subprocess.run(["sudo", "chmod", "777", ori_meta_file], capture_output=True, text=True)
    with open(ori_meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=4)


def get_latest_subdir(work_dir):
    dirs = [
        d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d)) and len(d) == 14 and d.isdigit()
    ]

    if not dirs:
        return None

    latest = max(dirs, key=lambda d: os.path.getmtime(os.path.join(work_dir, d)))
    return os.path.join(work_dir, latest)


def resolve_rl_meta_path(base_dir: str) -> str:
    for name in RL_META_CANDIDATES:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            print(f"Using RL meta file: {path}")
            return path
    tried = ", ".join(os.path.join(base_dir, name) for name in RL_META_CANDIDATES)
    raise FileNotFoundError(f"No RL meta file found. Tried: {tried}")


def main():
    base_dir = f"{sys.argv[1]}/{os.environ['GITHUB_RUN_ID']}/{sys.argv[2]}/{sys.argv[3]}"
    real_dir = get_latest_subdir(base_dir)
    new_meta = {"end": 10, "exp_dir": real_dir, "checkpoint_list": [f"{real_dir}/checkpoints/ckpt-step-10"]}
    if sys.argv[3] == "rl":
        update_meta(resolve_rl_meta_path(base_dir), new_meta)
    else:
        update_meta(f"{base_dir}/.xtuner", new_meta)


if __name__ == "__main__":
    main()
