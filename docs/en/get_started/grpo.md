```{important}
XTuner's RL (Reinforcement Learning) functionality is currently in Beta version. RL features are continuously being improved, and we welcome you to try it out and provide feedback.
```


# [Beta] RL: GRPO Training GSM8K



After experiencing SFT fine-tuning, let's further explore how to use XTuner for reinforcement learning (RL) training. We will use the GRPO (Group Relative Policy Optimization) algorithm as an example to introduce the basic process of RL training.

## Prepare Model

RL training can also be based on pre-trained models from Hugging Face. Let's take `Qwen3 8B` as an example, first download the model from Hugging Face:

```{code-block} bash
:caption: Download Qwen3 8B model

# Domestic users can use the huggingface mirror site, set environment variables before executing commands
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-8B --local-dir </path/to/qwen3-8B>

```

````{note}

The model path format requirements are consistent with SFT training. Please ensure the path points to the directory containing `config.json`.
````

## Prepare Dataset

On the basis of SFT fine-tuning, the reinforcement learning (RL) dataset needs to add evaluation information required by the reward model (Reward Model), such as `ground_truth` (standard answer). Let's take the `gsm8k` dataset as an example. XTuner provides a script to directly convert it from Hugging Face Hub to the required format.

**You can also directly use our provided example test dataset `tests/resource/gsm8k_train_example_data.jsonl`**

```{code-block} bash
:caption: Prepare dataset
# Domestic users can use the huggingface mirror site, set environment variables before executing commands
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download gsm8k --repo-type dataset --local-dir ./gsm8k_data

# Execute conversion script
python xtuner/v1/utils/convert_gsm8k.py --input-dir ./gsm8k_data --out-dir ./gsm8k
```

```{code-block} json
:caption: RL training dataset example

{
    "data_source": "openai/gsm8k",
    "prompt": [
        {
            "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after \"####\".",
            "role": "user"
        }
    ],
    "ability": "math",
    "reward_model": {
        "ground_truth": "72",
        "style": "rule"
    },
    "extra_info": {
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
        "index": 0,
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "split": "train"
    }
}
```

## Start GRPO Training

After preparing the dataset and model, you can start GRPO training through the command line. XTuner provides a dedicated RL training script. You just need to specify the model path, dataset path, and related training parameters:

```{code-block} bash
:caption: Start GRPO training

XTUNER_USE_FA3=1 XTUNER_USE_LMDEPLOY=1 python xtuner/v1/train/cli/grpo.py --model-path <model_path> --data-path tests/resource/gsm8k_train_example_data.jsonl

```

After executing the command, you will see log output similar to the following, indicating that RL training has been successfully started:

```{code-block} bash
:class: toggle
(DataFlow pid=387133) [XTuner][2025-09-07 09:49:29][INFO] Target batch size set to 128.
(DataFlow pid=387133) [XTuner][2025-09-07 09:49:29][INFO] Sample parameters set to n=1 top_k=0 top_p=1.0 temperature=1.0 repetition_penalty=1.0 presence_penalty=0.0 frequency_penalty=0.0 min_tokens=0 max_tokens=1024 stops=[] stop_token_ids=[] logprobs=0 skip_special_tokens=True do_sample=True.
rollout_controller for training samples:   0%|          | 0/128 [00:00<?, ?it/s]
rollout_controller for training samples:   9%|▉         | 12/128 [00:30<04:51,  2.51s/it]
rollout_controller for training samples:  27%|██▋       | 35/128 [00:31<01:22,  1.13it/s]
rollout_controller for training samples:  40%|███▉      | 51/128 [00:31<00:47,  1.63it/s]
rollout_controller for training samples:  47%|████▋     | 60/128 [00:45<00:51,  1.32it/s]
rollout_controller for training samples:  56%|█████▋    | 72/128 [00:51<00:40,  1.39it/s]
rollout_controller for training samples:  66%|██████▋   | 85/128 [00:53<00:27,  1.59it/s]
rollout_controller for training samples:  75%|███████▌  | 96/128 [00:53<00:17,  1.78it/s]
rollout_controller for training samples:  84%|████████▍ | 108/128 [01:02<00:11,  1.73it/s]
rollout_controller for training samples:  94%|█████████▍| 120/128 [01:06<00:04,  1.81it/s]
rollout_controller for training samples: 100%|██████████| 128/128 [01:11<00:00,  1.78it/s]
(DataFlow pid=387133) [XTuner][2025-09-07 09:50:41][INFO] Target batch size reached. Pausing env controller.
(DataFlow pid=387133) [XTuner][2025-09-07 09:50:41][INFO] send_samples_count: 128, unfinished_samples_count:0, finished_samples: 128, failed_samples: 0
[XTuner][RANK 0][2025-09-07 09:50:44][INFO] rollout_idx 1 finished, saved trajectories to work_dir/20250907094728/rollout_idx_1_trajectory.jsonl
[XTuner][RANK 0][2025-09-07 09:50:45][INFO] Training controller loaded
[XTuner][RANK 0][2025-09-07 09:50:47][INFO] Prepared 1024 training data batches
```

After training is completed, the model weights optimized by reinforcement learning will be saved in the working directory.

```{hint}
Want to learn more about detailed configuration and customization process of RL training?
- [Customize GRPO Training Using Python Code](../rl/tutorial/rl_grpo_trainer.rst)
- [RL Trainer Detailed Explanation](../api/rl_trainer.rst)
- [RL Training Configuration Detailed Explanation](../api/rl_config.rst)
```