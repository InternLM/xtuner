**注意：本文档的主要目标是详细说明如何根据 InternLM 仓库所提供的数据格式进行模型训练，而非如何训练 InternLM 模型。**

## 使用教程

### Step 1, 导出模板 config 文件

可以通过下列命令将名为 internlm_7b_full_intern_repo_dataset_template 的 config 导出至当前目录下：

```
xtuner copy-cfg internlm_7b_full_intern_repo_dataset_template .
```

### Step 2, 修改模板 config 文件

只需修改 Config 文件中上述接口对应部分即可。

```diff
...

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm-7b'

# Data
- dataset_folder = '/path/to/your/dataset'
+ dataset_folder = '/real/dataset/path'
max_length = 2048
pack_to_max_length = True
...
```

### Step 3, 启动训练

```
srun ${SRUN_ARGS} xtuner train internlm_7b_full_intern_repo_dataset_template_copy --launcher slurm --deepspeed deepspeed_zero1
```

## 数据集格式

[InternLM](https://github.com/InternLM/InternLM) 仓库所使用的训练数据集会被预先 token 化，格式如下所示：

```
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
```

其中，数值为负数的 tokens 在训练过程中不参与 loss 计算。
