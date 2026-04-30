import os
import sys

dist_packages_index = 0
for i, path in enumerate(sys.path):
    if path.endswith("dist-packages"):
        dist_packages_index = i
        break

if os.getenv('XTUNER_USE_LMDEPLOY', '').lower() in ['1', 'on', 'true']:
    lmdeploy_envs_dir = os.getenv('XTUNER_LMDEPLOY_ENVS_DIR', '/envs/lmdeploy')
    if lmdeploy_envs_dir not in sys.path:
        sys.path.insert(dist_packages_index, lmdeploy_envs_dir)

elif os.getenv('XTUNER_USE_SGLANG', '').lower() in ['1', 'on', 'true']:
    sglang_envs_dir = os.getenv('XTUNER_SGLANG_ENVS_DIR', '/envs/sglang')
    if sglang_envs_dir not in sys.path:
        sys.path.insert(dist_packages_index, sglang_envs_dir)
