# MMChat

### Installation

1. 安装 Pytorch (https://pytorch.org/get-started/locally/)
2. 安装 MMChat

```
pip install -r requirements.txt
pip install -e .
```

### TODO

##### 基础功能

- [x] MMEngine 加载 HF Dataset
- [x] MMEngine 加载 HF Model & Tokenizer
- [ ] DeepSpeed Config
- [ ] FSDP Config
- [ ] MMLU 评测
- [ ] PTH -> HF Checkpoint 转换脚本
- [ ] Attention Score 可视化

##### 算法

- [x] SFT
- [ ] Alpaca SFT Setting & 精度对齐
- [x] QLora SFT
- [ ] QLora SFT Setting & 精度对齐
- [ ] QLora RLHF
- [ ] Distill Finetune
