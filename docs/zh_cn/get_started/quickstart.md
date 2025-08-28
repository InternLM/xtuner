# 快速开始

本教程目的是希望用户能够在单卡且最小依赖下快速轻松上手 XTuner LLM 和 MLLM 微调，也可以用于环境验证和调试场景。

> 注意： 本教程模型并没有加载预训练权重，仅为随机初始化权重

## 极小 InternS1 多模态模型快速验证

相应配置文件为 `examples/v1/intern_s1_tiny_config.py`。 在训练过程中需要下载 `internlm/Intern-S1-mini` 的 tokenizer 相关文件。如果你的网络访问 huggingface.co 有问题，
可以提前手动下载，并手动修改上述配置文件中 `trainer.tokenizer_path` 的路径。

启动训练命令如下：

```shell
cd xtuner
# 目前单卡也需要采用 torchrun 启动
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 xtuner/v1/train/cli/sft.py --trainer-cfg-path examples/v1/intern_s1_tiny_config.py
```

以上任务可以在 24g 消费级显卡上运行。运行后会在当前路径下新建 `work_dirs` 文件夹存储权重和日志文件。

下面对配置文件进行简要说明：

