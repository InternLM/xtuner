## Quick Start

### step1: 环境准备

```
# 安装 pytorch
pip install torch==2.1.2+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装 xtuner rlhf 模块
git clone https://github.com/2581543189/xtuner.git
cd xtuner
git checkout rlhf
pip install .[rlhf]
```

### step2: 使用单引擎（huggingface）启动 rlhf 任务

```
# 启动任务
xtuner rlhf -c examples/rlhf/four_model_8gpu.py
```

### step3: 使用双引擎 (vllm + huggingface) 启动 rlhf 任务

```
# 安装 vllm
export VLLM_VERSION=0.3.3
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
pip uninstall xformers -y
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118
pip uninstall cupy-cuda12x -y
pip install cupy-cuda11x==12.1
python -m cupyx.tools.install_library --library nccl --cuda 11.x

# 启动任务
xtuner rlhf -c examples/rlhf/four_model_vllm_8gpu.py
```
