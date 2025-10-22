# Language Model Fine-tuning

After installing XTuner, let's try language model fine-tuning to get a taste of the simplest training startup method.

(sft-dataset)=
## Prepare Dataset

Before fine-tuning, you need to prepare the dataset. XTuner supports OpenAI format data by default, just organize the data into `jsonl` format:

```{code-block} json
:caption: jsonl format data example

[{"content": "Give three tips for staying healthy.\n", "role": "user"}, {"content": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.", "role": "assistant"}]
[{"content": "What are the three primary colors?\n", "role": "user"}, {"content": "The three primary colors are red, blue, and yellow.", "role": "assistant"}]

```

If you are training a GPT-OSS reasoning model, please read [GPT-OSS Chat Template Description](../pretrain_sft/tutorial/chat_template.md#gpt-oss-)


## Prepare Model

XTuner supports direct fine-tuning of models from Hugging Face. Let's take `Qwen3 8B` as an example, first download the pre-trained model from Hugging Face:


```{code-block} bash
:caption: Download Qwen3 8B model

# Domestic users can use the huggingface mirror site, set environment variables before executing commands
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-8B --local-dir </path/qwen3-8B>

```

````{note}

Note: The model path needs to be specific to the directory where the model files are located

```{code-block} bash
:caption: <span class="x-strong">Valid Model Path</span>

model-path/
├── config.json
├── model-00001-of-00005.safetensors
├── ...
```

Instead of a path with multiple versions like this:

```{code-block} bash
:caption: <del>Invalid Model Path<del>

models--Qwen--Qwen3-8B
├── blobs
├── refs
└── snapshots
```

If it is the above path structure, you need to specify a version number directory under snapshots, for example:

`models--Qwen--Qwen3-8B/snapshots/version_number`
````

## Start Fine-tuning

After preparing the dataset and model, you can start fine-tuning. XTuner provides a concise command line interface, just specify the model path, dataset path, and training parameters:

```{tip}
:class: margin

OOM? Try `--fsdp-config.cpu-offload`!

```
```{code-block} bash
:caption: Start fine-tuning training
torchrun --nproc-per-node 8  xtuner/v1/train/cli/sft.py  --load-from <model_path>  --chat_template qwen3 --dataset <dataset_path>  --total-step 100 --work-dir <target_work_directory>
```

After executing the command, you can see the following log:

```{code-block} bash
:class: toggle
[XTuner][RANK 2][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0578 lr: 0.000020 time: 4.9770 text_tokens: 4008.0 total_loss: 1.722 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 805.3 e2e_tgs: 796.1
[XTuner][RANK 5][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0641 lr: 0.000020 time: 4.9716 text_tokens: 4010.0 total_loss: 1.506 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 806.6 e2e_tgs: 796.3
[XTuner][RANK 6][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0617 lr: 0.000020 time: 4.9783 text_tokens: 4069.0 total_loss: 1.802 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 817.3 e2e_tgs: 807.3
[XTuner][RANK 7][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0614 lr: 0.000020 time: 4.9796 text_tokens: 4058.0 total_loss: 1.589 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 814.9 e2e_tgs: 805.0
[XTuner][RANK 1][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0571 lr: 0.000020 time: 4.9848 text_tokens: 3929.0 total_loss: 1.623 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 788.2 e2e_tgs: 779.3
[XTuner][RANK 3][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0600 lr: 0.000020 time: 4.9837 text_tokens: 4077.0 total_loss: 1.686 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 818.1 e2e_tgs: 808.3
[XTuner][RANK 4][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0542 lr: 0.000020 time: 4.9981 text_tokens: 3931.0 total_loss: 1.779 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 786.5 e2e_tgs: 778.1
[XTuner][RANK 0][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0674 lr: 0.000020 time: 4.9857 text_tokens: 4044.0 total_loss: 1.764 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 811.1 e2e_tgs: 800.3
[XTuner][RANK 2][2025-08-29 09:17:52][INFO] Step 2/100 data_time: 0.0516 lr: 0.000040 time: 0.8883 text_tokens: 4037.0 total_loss: 1.592 reduced_llm_loss: 1.606 max_memory: 18.02 GB reserved_memory: 22.20 GB grad_norm: 12.398 tgs: 4544.6 e2e_tgs: 1346.2
[XTuner][RANK 5][2025-08-29 09:17:52][INFO] Step 2/100 data_time: 0.0442 lr: 0.000040 time: 0.8948 text_tokens: 4049.0 total_loss: 1.620 reduced_llm_loss: 1.606 max_memory: 18.02 GB reserved_memory: 22.20 GB grad_norm: 12.398 tgs: 4524.8 e2e_tgs: 1348.5
[XTuner][RANK 1][2025-08-29 09:17:52][INFO] Step 2/100 data_time: 0.0438 lr: 0.000040 time: 0.8899 text_tokens: 4031.0 total_loss: 1.367 reduced_llm_loss: 1.606 max_memory: 18.02 GB reserved_memory: 22.20 GB grad_norm: 12.398 tgs: 4529.9 e2e_tgs: 1331.8
```

```{tip}
:class: margin

There's also a `.xtuner` file in the working directory. Go check what's written in it?
```

Compared with the verification log in [Quick Start](./installation.md), this initial loss is significantly lower because we loaded pre-trained model weights and a real tokenizer. After training is completed, you can see the corresponding model weights saved in the working directory.


```{hint}
Want to learn more about training parameters and configuration options? Check out these tutorials:
- [Configuration File Start Training](../pretrain_sft/tutorial/config.md)
- [Python Code Start Training](../pretrain_sft/tutorial/llm_trainer.md).
```