# How to use XTuner in HuggingFace training pipeline

## Quick run

1. step in `examples`

   ```shell
   cd ./examples
   ```

2. run training scripts

   ```shell
   # qlora-training internlm-7b with alpaca dataset
   python train_qlora_hf.py --model_name_or_path internlm/internlm-7b --dataset_name_or_path tatsu-lab/alpaca
   ```

   `--model_name_or_path`: specify the model name or path to train.

   `--dataset_name_or_path`: specify the dataset name or path to use.

## How to customize your experiment

XTuner APIs are compatible with the usage of HuggingFace's transformers.
If you want to customize your experiment, you just need to pass in your hyperparameters like HuggingFace.

```
# training example
python train_qlora_hf.py \
    # custom training args
    --model_name_or_path internlm/internlm-7b \
    --dataset_name_or_path tatsu-lab/alpaca \
    # HuggingFace's default training args
    --do_train = True
    --per_device_train_batch_size = 1
    --learning_rate = 2e-5
    --save_strategy = 'epoch'
    --lr_scheduler_type = 'cosine'
    --logging_steps = 1
```
