
source ./zdev/env.sh
source $(conda info --base)/etc/profile.d/conda.sh
conda activate py312-pt28-raw

pytest --durations=20 \
  tests/rl/test_agent_loop_utils.py \
  tests/rl/test_multi_task_agent_loop_manager.py \
  tests/rl/test_producer.py \
  tests/rl/test_rl_colocate_trainer.py \
  tests/rl/test_rl_disaggregated_trainer.py \
  tests/rl/test_agent_loop.py::TestAgentLoop::test_gsm8k_agent_loop_manager
