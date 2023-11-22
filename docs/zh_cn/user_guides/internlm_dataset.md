## 数据集格式

Internlm 训练数据集是已经被 tokenized 过的，格式如下所示：

```
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
```

其中，数值为负数的 tokens 在训练过程中不参与 loss 计算。

## 接口介绍

为了在 XTuner 中训练 InternLM-Chat 模型，需要将原数据格式转换为 XTuner 标准数据集格式。处理 InternLM 格式数据集的核心函数如下：

```python
def process(dataset_folder=None,
            cached_folder=None,
            split='train',
            pack_to_max_length=False,
            max_length=2048,
            shuffle_before_pack=True,
            map_num_proc=32):
    ......
```

其中：

1. `dataset_folder`：表示训练数据集所在路径，路径下的所有以 `.bin` 结尾的文件都会被当做训练数据。由于处理后的数据会被缓存下来，因此**只有第一次处理原始数据的时候才需要提供 dataset_folder 这一字段**。
2. `cached_folder`：表示处理后的数据缓存至 cached_folder 文件夹下。
3. `split`：通过hf datasets 读取的数据集通常是一个 DatasetDict ，需要通过split变量拿到对应的Dataset，一般使用默认值 train 即可
4. `pack_to_max_length`：是否将多条数据拼接为一条数据进行训练。
5. `max_length`：表示数据处理过程会将多条训练数据 pack 成一条长度为max_length 的数据。只有当pack_to_max_length=True时生效。
6. `shuffle_before_pack`：在pack前是否对数据集进行shuffle，一般使用默认的True即可。只有当pack_to_max_length=True时生效。
7. `map_num_proc`：启用多进程进行数据处理，可根据情况增加 map_num_proc 的数值，集群上可以设为 96 。

## 使用教程

### Step 1, 导出模板 config 文件

可以通过下列命令将名为 internlm_7b_full_internlm_dataset_template 的 config 导出至当前目录下：

```
xtuner copy-cfg internlm_7b_full_internlm_dataset_template .
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
srun ${SRUN_ARGS} xtuner train internlm_7b_full_internlm_dataset_template --launcher slurm --deepspeed deepspeed_zero3
```
