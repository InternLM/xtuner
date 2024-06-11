# Llama3 8B

## Install

```bash
# Install the latest xtuner
pip install -U 'xtuner[deepspeed]'

# install the latest transformers
pip install -U transformers
```

## QLoRA Fine-tune

QLoRA only need a single A100-80G

```bash
xtuner train llama3_8b_instruct_qlora_alpaca_e3
```

## Full Parameter Fine-tune

Full parameter fine-tune Llama3 8B in 8k context only requires 2 * A100-80G

### torchrun

```bash
NPROC_PER_NODE=${GPU_NUM} xtuner train llama3_8b_instruct_full_alpaca_e3 --deepspeed deepspeed_zero2
```

### slurm

```bash
srun ${SRUN_ARGS} xtuner train llama3_8b_instruct_full_alpaca_e3 --launcher slurm --deepspeed deepspeed_zero3
```

### Speed

|   Model   | Sequence Length | GPU Number |  ZeRO  | Sequence Parallel | Tokens per Second | TFLOPs |
| :-------: | :-------------: | :--------: | :----: | :---------------: | :---------------: | :----: |
| Llama3 8B |       8k        |     2      | ZeRO-3 |         2         |      1037.0       |  76.8  |
| Llama3 8B |       8k        |     4      | ZeRO-3 |         1         |      2331.3       | 172.6  |
| Llama3 8B |       8k        |     8      | ZeRO-3 |         1         |      2771.2       | 205.1  |

|   Model   | Sequence Length | GPU Number |  ZeRO  | Sequence Parallel | Tokens per Second | TFLOPs |
| :-------: | :-------------: | :--------: | :----: | :---------------: | :---------------: | :----: |
| Llama3 8B |       8k        |     8      | ZeRO-3 |         1         |      2771.2       | 205.1  |
| Llama3 8B |       16k       |     8      | ZeRO-3 |         2         |      2320.7       | 191.7  |
| Llama3 8B |       32k       |     8      | ZeRO-3 |         4         |      1870.2       | 186.6  |
| Llama3 8B |       64k       |     8      | ZeRO-3 |         8         |      1356.4       | 182.0  |
| Llama3 8B |      128k       |     8      | ZeRO-3 |         8         |       875.7       | 177.7  |
