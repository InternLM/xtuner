
cd $(dirname $0)
source ./zdev/env.sh
source $(conda info --base)/etc/profile.d/conda.sh
conda activate py312-pt28-raw

pytest -v -s tests/datasets/test_preset_pack_dataset.py