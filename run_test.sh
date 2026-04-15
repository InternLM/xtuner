
source ./zdev/env.sh
source $(conda info --base)/etc/profile.d/conda.sh
conda activate py312-pt28

pytest -qq tests/rl/test_agent_loop.py \
  tests/rl/test_multi_task_agent_loop_manager.py
