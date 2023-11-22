**注意：本文档的主要目标是详细说明如何根据 InternLM 仓库所提供的数据格式进行模型训练，而非如何训练 InternLM 模型。**

## 数据集格式

[InternLM](https://github.com/InternLM/InternLM) 仓库所使用的训练数据集会被预先 token 化，格式如下所示：

```
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
```

其中，数值为负数的 tokens 在训练过程中不参与 loss 计算。

## 接口介绍

为了在 XTuner 中使用 InternLM 格式的数据进行训练，需要将原数据格式转换为 XTuner 标准数据集格式。处理 InternLM 格式数据集的核心函数如下：

```python
def process(dataset_folder,
            split='train',
            pack_to_max_length=False,
            max_length=2048,
            shuffle_before_pack=True,
            map_num_proc=32):
    ......
```

其中：

1. `dataset_folder`：表示训练数据集所在路径，路径下的所有以 `.bin` 结尾的文件都会被当做训练数据。
2. `split`：通过hf datasets 读取的数据集通常是一个 DatasetDict ，需要通过split变量拿到对应的Dataset，一般使用默认值 train 即可
3. `pack_to_max_length`：是否将多条数据拼接为一条数据进行训练。
4. `max_length`：表示数据处理过程会将多条训练数据 pack 成一条长度为max_length 的数据。只有当pack_to_max_length=True时生效。
5. `shuffle_before_pack`：在pack前是否对数据集进行shuffle，一般使用默认的True即可。只有当pack_to_max_length=True时生效。
6. `map_num_proc`：启用多进程进行数据处理，可根据情况增加 map_num_proc 的数值。

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
srun ${SRUN_ARGS} xtuner train internlm_7b_full_intern_repo_dataset_template --launcher slurm --deepspeed deepspeed_zero3
```
