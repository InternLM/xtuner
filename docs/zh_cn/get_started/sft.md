# 微调 Dense & MoE

本教程为标准的微调教程，加载权重后对模型进行训练，然后自动保存为 HF 格式模型。

## 微调 Qwen3 大模型

## 微调 Qwen3-MoE 大模型

## 微调 Intern-S1-Mini 科学多模态模型

相应配置文件为 `examples/v1/sft_intern_s1_mini_config.py`。 在训练过程中需要下载 `internlm/Intern-S1-mini` 的模型文件。如果你的网络访问 huggingface.co 有问题，
可以提前手动下载，并手动修改上述配置文件中 `hf_model_path` 的路径。

启动训练命令如下：

```shell
cd xtuner
torchrun --nproc-per-node=8 xtuner/v1/train/cli/sft.py --trainer-cfg-path examples/v1/sft_intern_s1_mini_config.py
```

运行后会在当前路径下新建 `work_dirs` 文件夹存储权重和日志文件。 为了方便大家能方便构建自己的数据和配置，下面对 `examples/v1/sft_intern_s1_mini_config.py` 进行说明。

### 构建模型配置

```python
vision_cfg = InternS1VisionConfig()
projector_cfg = InternS1ProjectorConfig()
llm_cfg = Qwen3_8BConfig(vocab_size=153216)
model_cfg = InternS1Config(vision_config=vision_cfg,
                           text_config=llm_cfg,
                           projector_config=projector_cfg,
                           freeze_vision=True, # 考虑 freeze vit
                           freeze_projector=True # 考虑 freeze projector
                           )
```

InternS1 包括 3 个核心模块，分别对于 3 个配置类：
- `InternS1VisionConfig`： 视觉编码器配置
- `InternS1ProjectorConfig`： 视觉-语言投影层配置
- `Qwen3_8BConfig`： 语言模型配置

请注意： 如果你想完全加载 HF 官方权重，则默认参数都不能修改，否则后续会出现权重加载不上情况。

### 构建数据配置

```python
sample_max_length = 8192 # 单条样本的最大长度，超过会被截断，并且会有警告输出
pack_max_length = 16384 # 训练一次 iter 所能包含的最大长度，pack 机制会尽可能将多条样本拼接在一起，减少 padding
# 如果你的显存不够，可以适当调小上述两个参数，但是请确保 sample_max_length <= pack_max_length

# dataset 构建也采用配置方式，每一类 jsonl 数据作为一个 item
dataset_config = [
    {
        "dataset": DatasetConfig(name='pure_text', # 数据别名
                                 # 标注文件路径，可以是单个 jsonl 也可以是文件夹，会自动遍历当前文件夹下所有 jsonl 文件
                                 anno_path='tests/resource/mllm_sft_text_example_data.jsonl', # 纯文本数据
                                 sample_ratio=2.0, # 数据采样比例，这里是重复 2 遍，可以是小数
                                 class_name='VLMJsonlDataset'), # 对应的 dataset 类名
        # 一个 dataset 要配一个对应的 tokenizer fun 函数用于处理 dataset 输出的单条 item 数据
        "tokenize_fn": InternS1TokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
    {
        "dataset": DatasetConfig(name='media', # 数据别名
                                 anno_path='tests/resource/mllm_sft_media_example_data.jsonl', # 多模态数据
                                 media_root='tests/',
                                 sample_ratio=10.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": InternS1TokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
]
# dataloader 配置
dataloader_config = DataloaderConfig(pack_max_length=pack_max_length, 
                                     pack_level="expand_soft", # pack 样本有 2 种策略，默认选择更高效的 expand_soft 策略
                                     collator='sft_vllm_collator')
```

通过上述灵活的配置组合方式，用户可以轻松配置各类数据集，并且控制各自的采样比例。

### 构建训练配置

```python
optim_cfg = AdamWConfig(lr=1e-6, foreach=False) # 不同模块的 device mesh 有差别，foreach 必须是 False
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0)
```

### 构建 Trainer 配置

```python
trainer = TrainerConfig(
    load_from=hf_model_path, # 如果是微调模式，必须指定，否则会重头训练
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    tokenizer_path=hf_model_path,
    # 全局 batch size
    # 假设是 8 卡训练，那么每张卡的 forward shape 是 (1, pack_max_length)，梯度累加次数是 1
    # 假设是 4 卡训练，那么每张卡的 forward shape 是 (1, pack_max_length)，梯度累加次数是 2 (自动折算)
    global_batch_size=8, 
    epoch_num=1,
    chunked_loss=True, # 可以显著减少显存占用，推荐总是开启
    work_dir='work_dirs'
)
```

