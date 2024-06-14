# DeepSeek V2

## Install

```bash
# Git clone the latest xtuner
git clone https://github.com/InternLM/xtuner.git

# Install the latest xtuner
cd xtuner
pip install -e '.[all]'

# Mixtral requires flash-attn
pip install flash-attn

# install the latest transformers
pip install -U transformers
```

## Full Parameter Fine-tune

Full parameter fine-tune DeepSeek V2 236B needs at least 64 A100-80G. The full-tuned model will be saved to `${WORK_DIRS}/hf_model` by `HFCheckpointHook`.

### slurm

Note: `$PARTITION` means the virtual partition of slurm.

```bash
srun -p $PARTITION --job-name=mixtral --nodes=8 --gres=gpu:8 --ntasks-per-node=8 xtuner train deepseek_v2_chat_full_alpaca_e3 --deepspeed deepspeed_zero3 --launcher slurm
```

### torchrun

Note: `$NODE_0_ADDR` means the ip address of the node_0 machine.

```bash
# excuete on node 0
NPROC_PER_NODE=8 NNODES=8 PORT=29600 ADDR=$NODE_0_ADDR NODE_RANK=0 xtuner train deepseek_v2_chat_full_alpaca_e3 --deepspeed deepspeed_zero3 --launcher pytorch

# excuete on node 1
NPROC_PER_NODE=8 NNODES=8 PORT=29600 ADDR=$NODE_0_ADDR NODE_RANK=1 xtuner train deepseek_v2_chat_full_alpaca_e3 --deepspeed deepspeed_zero3 --launcher pytorch

# excuete on node 2, 3, ..., 7
```

### Speed

128 * A100 80G:

|         Model          | Sequence Length | Use Varlen Attn | Sequence Parallel World Size | Tokens per Second |
| :--------------------: | :-------------: | :-------------: | :--------------------------: | :---------------: |
|     deepseek v2 hf     |       8k        |      False      |              1               |        60         |
| **deepseek v2 XTuner** |     **8k**      |    **False**    |            **1**             |   **120 (2x)**    |
|     deepseek v2 hf     |       8k        |      True       |              1               |        60         |
| **deepseek v2 XTuner** |     **8k**      |    **True**     |            **1**             |  **130 (2.2x)**   |
|     deepseek v2 hf     |       16k       |      False      |              1               |        OOM        |
| **deepseek v2 XTuner** |     **16k**     |    **False**    |            **1**             |      **148**      |
|     deepseek v2 hf     |       16k       |      True       |              1               |        95         |
| **deepseek v2 XTuner** |     **16k**     |    **True**     |            **1**             |  **180 (1.9x)**   |
