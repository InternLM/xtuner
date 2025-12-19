import json
import os
import subprocess


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
    dirs = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]

    if not dirs:
        return None

    latest = max(dirs, key=lambda d: os.path.getmtime(os.path.join(work_dir, d)))
    return os.path.join(work_dir, latest)



base_dir = (
    f"/mnt/shared-storage-user/llmrazor-share/qa-llm-cicd/test_output/{os.environ['GITHUB_RUN_ID']}/qwen3-sft-ep8/sft"
)
real_dir = get_latest_subdir(base_dir)
new_meta = {"end": 10, "exp_dir": real_dir, "checkpoint_list": [f"{real_dir}/checkpoints/ckpt-step-10"]}
update_meta(f"{base_dir}/.xtuner", new_meta)
