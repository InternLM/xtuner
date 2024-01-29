## 功能说明

### 已支持的功能

- 保存 DeepSpeed Checkpoint 至 CEPH
- 从 Ceph 上的 DeepSpeed Checkpoint 续训
- `pth_to_hf` 支持 Ceph 上的 DeepSpeed Checkpoint

### 暂不支持的功能

- 训练时从 Ceph 加载 Huggingface 模型， 与 `zero3` 加载权重冲突
- HuggingFace `save_pretrained` 保存至 Ceph， 逻辑过于复杂，没办法 patch

## 使用说明

#### 1. 验证 ceph 环境

使用前需确保 `petrel sdk` 可用，并且要使用的 Ceph bucket 存在且可用

验证 `aws` 命令行工具

```bash
# 验证 aws 命令行工具
aws s3 ls $YOUR_BUCKET
```

验证 `petrel sdk`

```python
bucket = 's3://xxx'

from mmengine import get_file_backend
backend = get_file_backend(bucket)

for f in backend.list_dir_or_file(bucket):
    print(f)
```

#### 2. 训练时保存 Checkpoint 至 Ceph

`XTuner` 根据环境变量 `DS_CEPH_DIR` 来判断是否将 checkpoint 保存至 ceph

```bash
DS_CEPH_DIR=s3://xxxx srun ${SRUN_ARGS} xtuner train $CONFIG --launcher slurm
```

#### 3. 从 Ceph 上的 Checkpoint 续训

Resume 时，要填写 checkpoint 在 ceph 上的完整路径

```bash
DS_CEPH_DIR=s3://xxxx srun ${SRUN_ARGS} xtuner train $CONFIG --launcher slurm --resume s3://xxx/yyy/epoch_x.pth
```

#### 4. 将 Ceph 上的 Checkpoint 转换为 HF 模型

不支持 `$HF_DIR` 为 ceph 路径

由于 Checkpoint 中存储了优化器状态，加载比较耗时，对于 ZeRO 1&2 可以直接加载 checkpoint 中的 `model_states.pt` 文件加速转换过程；ZeRO 3 必须先加载整个 checkpoint

```bash
srun ${SRUN_ARGS} xtuner convert pth_to_hf $CONFIG s3://xxx/yyy/epoch_x.pth $HF_DIR

```
