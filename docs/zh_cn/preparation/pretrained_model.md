# 准备预训练模型权重

本节将以下载 internlm2-chat-7b 为例，介绍如何快速下载预训练模型的权重。

## 方法 1：利用 `snapshot_download`

`huggingface_hub.snapshot_download` 支持下载特定的 HuggingFace Hub 模型权重，并且允许多线程。您可以利用下列代码并行下载模型权重：

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id='internlm/internlm2-chat-7b', max_workers=20)
```

- ModelScope?

  - ```python
    from modelscope import snapshot_download

    snapshot_download(model_id='Shanghai_AI_Laboratory/internlm2-chat-7b')
    ```

  - 注：`modelscope.snapshot_download` 不支持多线程并行下载。

## 方法 2：利用 Git LFS

HuggingFace 和 ModelScope 的远程模型仓库就是一个由 Git LFS 管理的 Git 仓库。因此，我们可以利用 `git clone` 完成权重的下载：

```shell
git lfs install
# For HuggingFace
git clone https://huggingface.co/internlm/internlm2-chat-7b
# For ModelScope
git clone https://www.modelscope.cn/Shanghai_AI_Laboratory/internlm2-chat-7b.git
```

## 方法 3：利用 `AutoModelForCausalLM.from_pretrained`

`AutoModelForCausalLM.from_pretrained` 在初始化模型时，将尝试连接远程仓库并自动下载模型权重。因此，您可以执行下列代码下载您的模型权重：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('internlm/internlm2-chat-7b', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm2-chat-7b', trust_remote_code=True)
```

此时模型将会下载至 HuggingFace 的 cache 路径中（默认为`~/.cache/huggingface`）。

- ModelScope?

  - 如果您期望从 ModelScope 下载模型，可以使用 `modelscope` 库所提供的模型接口。

  - ```python
    from modelscope import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained('Shanghai_AI_Laboratory/internlm2-chat-7b', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('Shanghai_AI_Laboratory/internlm2-chat-7b', trust_remote_code=True)
    ```

  - 此时模型将会下载至 ModelScope 的 cache 路径中（默认为`~/.cache/modelscope/hub`）
