# How to use MMChat in HuggingFace & DeepSpeed training pipelines

## Quick run

1. step in `examples`

   ```shell
   cd ./examples
   ```

2. run training scripts

   ```shell
   # HuggingFace training pipeline
   bash run_train_alpaca-qlora_with_mmchat_hf.sh

   # DeepSpeed training pipeline
   bash run_train_alpaca-qlora_with_mmchat_deepspeed.sh
   ```

3. (optional) whether to use `qlora` / `lora` or not

   you can choose whether to use `qlora` / `lora` in training just by change the setting in scripts.

   ```
   # case 1. use qlora
   --use_qlora True
   --use_lora False

   # case 2. use lora
   --use_qlora False
   --use_lora True

   # case 3. neither
   --use_qlora False
   --use_lora False
   ```

## Training pipeline

If you want to use mmchat for efficient finetuning in your original training pipelines, you just need to change the implement of building `model` and `dataloader`, reserve other parts. Thus you can quickly fine-tune various models with various datasets by changing the relevant configs.

## How to build model with MMChat for QLoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from mmchat.models import SupervisedFinetune

model_name_or_path = 'internlm/internlm-7b'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          trust_remote_code=True)
# if use lora, `quantization_config = None`
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4')

# if neither use QLoRA or LoRA, `lora_config = None`
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias='none',
    task_type='CAUSAL_LM')

# build base llm model
llm = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=quantization_config,
    trust_remote_code=True
)

# build model with MMChat
model = SupervisedFinetune(llm=llm, lora=lora_config, tokenizer=tokenizer)

```

## How to build train_dataloader

```python
from .data_utils import get_train_dataloader

dataset_cfg_path = '../configs/_base_/datasets/alpaca.py'

# need to pass in `tokenizer` previously prepared
train_dataloader = get_train_dataloader(dataset_cfg_path, tokenizer)

# (optional) you can get also dataset or collate_fn by train_dataloader
train_dataset = train_dataloader.dataset
data_collator = train_dataloader.collate_fn
```
