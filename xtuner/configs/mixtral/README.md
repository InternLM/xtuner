# Mixtral 8x7b

## Install

```bash
# Mixtral requires the latest version of transformers.
pip install git+https://github.com/huggingface/transformers.git

# Mixtral requires flash-attn
pip install flash-attn

# install xtuner and deepspeed
pip install -U 'xtuner['deepspeed']'
```

## Chat Template

Due to the lack of official dialogue templates from Mixtral, we use InternLM's dialogue templates for its SFT fine-tuning.

## QLoRA Finetune

QLoRA only need a single A100-80G

```bash
xtuner train mixtral_8x7b_qlora_oasst1_internlm_template_e3 --deepspeed deepspeed_zero2
```

## Full Parameter Finetune

Full parameter finetune needs 32 A100-80G

### slurm

Note: `$PARTITION` means the virtual partition of slurm.

```bash
srun -p $PARTITION --job-name=mixtral --nodes=4 --gres=gpu:8 --ntasks-per-node=8 xtuner train mixtral_8x7b_full_oasst1_internlm_template_e3 --deepspeed deepspeed_zero3 --launcher slurm
```

### torchrun

Note: `$NODE_0_ADDR` means the ip address of the node_0 machine.

```bash
# excuete on node 0
NPROC_PER_NODE=8 NNODES=4 PORT=29600 ADDR=$NODE_0_ADDR NODE_RANK=0 xtuner train mixtral_8x7b_full_oasst1_internlm_template_e3 --deepspeed deepspeed_zero3

# excuete on node 1
NPROC_PER_NODE=8 NNODES=4 PORT=29600 ADDR=$NODE_0_ADDR NODE_RANK=1 xtuner train mixtral_8x7b_full_oasst1_internlm_template_e3 --deepspeed deepspeed_zero3

# excuete on node 2
NPROC_PER_NODE=8 NNODES=4 PORT=29600 ADDR=$NODE_0_ADDR NODE_RANK=2 xtuner train mixtral_8x7b_full_oasst1_internlm_template_e3 --deepspeed deepspeed_zero3

# excuete on node 3
NPROC_PER_NODE=8 NNODES=4 PORT=29600 ADDR=$NODE_0_ADDR NODE_RANK=3 xtuner train mixtral_8x7b_full_oasst1_internlm_template_e3 --deepspeed deepspeed_zero3
```
