# Qwen 110B

## Install

```bash
# Install the latest xtuner
pip install -U 'xtuner[deepspeed]'

# We recommend installing flash_attn
# pip install flash-attn

# install the latest transformers
pip install -U transformers
```

## QLoRA Fine-tune

Training Qwen 110B with 32k context capability requires only 2 * A100 80G.

```bash
xtuner train xtuner/configs/qwen/qwen1_5/qwen1_5_110b_chat/qwen1_5_110b_chat_qlora_alpaca_e3_16k_2gpus.py --deepspeed deepspeed_zero3
```

<div align=center>
  <img src="https://github.com/InternLM/xtuner/assets/41630003/48e4b6e3-1bcd-4349-90f0-dbbbc0f1cee7" style="width:80%">
</div>
