# Mixtral 8x7B

## Install

```bash
# Install the latest xtuner
pip install -U 'xtuner[deepspeed]'

# Mixtral requires the latest version of transformers.
pip install git+https://github.com/huggingface/transformers.git

# Mixtral requires flash-attn
pip install flash-attn

# install the latest transformers
pip install -U transformers
```

## QLoRA Fine-tune

QLoRA only need a single A100-80G

```bash
xtuner train mixtral_8x7b_instruct_qlora_oasst1_e3 --deepspeed deepspeed_zero2
```

## Full Parameter Fine-tune

Full parameter fine-tune needs 16 A100-80G

### slurm

Note: `$PARTITION` means the virtual partition of slurm.

```bash
srun -p $PARTITION --job-name=mixtral --nodes=2 --gres=gpu:8 --ntasks-per-node=8 xtuner train mixtral_8x7b_instruct_full_oasst1_e3 --deepspeed deepspeed_zero3 --launcher slurm
```

### torchrun

Note: `$NODE_0_ADDR` means the ip address of the node_0 machine.

```bash
# excuete on node 0
NPROC_PER_NODE=8 NNODES=2 PORT=29600 ADDR=$NODE_0_ADDR NODE_RANK=0 xtuner train mixtral_8x7b_instruct_full_oasst1_e3 --deepspeed deepspeed_zero3

# excuete on node 1
NPROC_PER_NODE=8 NNODES=2 PORT=29600 ADDR=$NODE_0_ADDR NODE_RANK=1 xtuner train mixtral_8x7b_instruct_full_oasst1_e3 --deepspeed deepspeed_zero3
```
