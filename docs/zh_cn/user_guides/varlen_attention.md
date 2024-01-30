# 在训练大语言模型时使用变长注意力 (Variable Length Attention)

## 使用教程

### Step 1, 安装 flash_attn

XTuner 中实现的变长注意力需要依赖 Flash Attention 2，可通过以下命令安装：

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### Step 2, 列出候选模型名字

XTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看：

```bash
xtuner list-cfg -p internlm
```

`-p` 为模糊查找，若想训练其他模型，可以修改 `internlm` 为 XTuner 支持的其他模型名称。

### Step 3, 复制 config 文件

导出需要使用的 config ：

```bash
xtuner copy-cfg ${CONFIG_NAME} ${SAVE_DIR}
```

例如通过下列命令将名为 `internlm_7b_full_oasst1_e3` 的 config 导出至当前目录下：

```bash
xtuner copy-cfg internlm_7b_full_oasst1_e3 .
```

### Step 4, 修改 config 文件

将 Step 3 复制得到的 config 文件中的 `use_varlen_attn` 属性由 False 改为 True 即可激活变长注意力训练机制：

```diff
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm-7b'
- use_varlen_attn = False
+ use_varlen_attn = True
...
```

**需要注意，当设置 `use_varlen_attn = True` 后，请确保 `batch_size` 被设置为 1，且 `pack_to_max_length` 被设置为 True。**

### Step 5, 开始训练

```
xtuner train ${CONFIG_NAME_OR_PATH}
```

例如，我们可以基于 Step 4 中修改得到的 `internlm_7b_full_oasst1_e3_copy.py` 进行训练：

```bash
# On a single GPU
xtuner train internlm_7b_full_oasst1_e3_copy.py --deepspeed deepspeed_zero1
# On multiple GPUs
(DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm_7b_full_oasst1_e3_copy.py --deepspeed deepspeed_zero1
(SLURM) srun ${SRUN_ARGS} xtuner train internlm_7b_full_oasst1_e3_copy.py --launcher slurm --deepspeed deepspeed_zero1
```

- `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 来优化训练过程。若未安装 DeepSpeed ，可通过 `pip install deepspeed>=0.12.3` 进行安装。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 等。如果用户期望关闭此功能，请直接移除此参数。

### Step 6, 模型转换

将保存的 PTH 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 HuggingFace 模型：

```
xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH} ${SAVE_PATH}
```

对应上面的例子，模型转换脚本为：

```
xtuner convert pth_to_hf internlm_7b_full_oasst1_e3_copy.py ${PTH} ${SAVE_PATH}
```

其中 `${PTH}` 为训练权重保存的路径，若未指定，默认保存在 `./work_dirs/internlm_7b_full_oasst1_e3_copy` 路径下。

## 变长注意力训练策略原理

<div align="center">
  <img src="https://github.com/InternLM/InternLM/assets/41630003/7e0c6a02-a970-4bd3-a10b-8341720bf654" width="600"/>
  <br /><br />
</div>

假设一条由若干条*短数据*拼接成的数据长度为 4096 。若不使用变长注意力机制，在计算 attention 阶段，每个 token 会关注全部 4096 个 tokens ，如上图左侧所示。当使用变长注意力机制时，计算 attention 阶段每个 token 仅会关注他所在的那条*短数据*中所有的 tokens，如上图右侧所示。
