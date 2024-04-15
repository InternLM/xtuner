# Cohere 8x7B

## Install

```bash
# Install the latest xtuner
pip install -U 'xtuner[deepspeed]'

# Cohere requires the latest version of transformers.
pip install git+https://github.com/huggingface/transformers.git

# Sequence parallel requires flash-attn
pip install flash-attn
```

## Full Parameter Fine-tune

Full parameter fine-tune needs 64 A100-80G

### slurm

Note: `$PARTITION` means the virtual partition of slurm.

```bash
srun -p $PARTITION --job-name=Cohere --nodes=8 --gres=gpu:8 --ntasks-per-node=8 xtuner train cohere_100b_128k_sp32 --deepspeed deepspeed_zero3 --launcher slurm
```

### torchrun

Note: `$NODE_0_ADDR` means the ip address of the node_0 machine.

```bash
# excuete on node 0
NPROC_PER_NODE=8 NNODES=8 PORT=29600 ADDR=$NODE_0_ADDR NODE_RANK=0 xtuner train cohere_100b_128k_sp32 --deepspeed deepspeed_zero3

# excuete on node 1
NPROC_PER_NODE=8 NNODES=8 PORT=29600 ADDR=$NODE_0_ADDR NODE_RANK=1 xtuner train cohere_100b_128k_sp32 --deepspeed deepspeed_zero3
```

### Speed

16 * A100 80G:

|    Model    | Sequence Length | GPUs Number | Sequence Parallel World Size | Tokens per Second | TFLOPs |
| :---------: | :-------------: | :---------: | :--------------------------: | :---------------: | :----: |
| Cohere_100b |      128k       |     64      |              32              |       97.3        | 173.4  |
| Cohere_100b |      128k       |     128     |              16              |       102.1       | 182.7  |
| Cohere_100b |      128k       |     256     |              16              |       101.3       | 181.3  |
