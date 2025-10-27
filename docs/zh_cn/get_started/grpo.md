```{important}
XTuner 的 RL（强化学习）功能目前为 Beta 版本，RL功能特性持续完善中，欢迎试用并反馈问题。
```


# [Beta] RL: GRPO训练GSM8K



在体验了 SFT 微调后，让我们进一步探索如何使用 XTuner 进行强化学习（RL）训练。我们将以 GRPO（Group Relative Policy Optimization）算法为例，介绍 RL 训练的基本流程。

## 准备模型

RL 训练同样可以基于 Hugging Face 上的预训练模型进行。我们以 `Qwen3 8B` 为例，先从 Hugging Face 下载模型：

```{code-block} bash
:caption: 下载 Qwen3 8B 模型

# 国内用户可以使用 huggingface 镜像站点，在执行命令之前设置环境变量
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-8B --local-dir </path/to/qwen3-8B>

```

````{note}

模型路径的格式要求与 SFT 训练一致，请确保路径指向包含 `config.json` 的目录。
````

## 准备数据集

强化学习（RL）的数据集在SFT微调的基础上，需要增加奖励模型（Reward Model）所需的评估信息，如 `ground_truth`（标准答案）。我们以 `gsm8k` 数据集为例，XTuner 提供了脚本将其从 Hugging Face Hub 直接转换为符合要求的格式。

**您也可以直接使用我们提供的示例测试数据集 `tests/resource/gsm8k_train_example_data.jsonl `**

```{code-block} bash 
:caption: 准备数据集
# 国内用户可以使用 huggingface 镜像站点，在执行命令之前设置环境变量
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download gsm8k --repo-type dataset --local-dir ./gsm8k_data

# 执行转换脚本
python xtuner/v1/utils/convert_gsm8k.py --input-dir ./gsm8k_data --out-dir ./gsm8k
```

```{code-block} json
:caption: RL 训练数据集示例

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

## 启动 GRPO 训练

准备好数据集和模型后，即可通过命令行启动 GRPO 训练。XTuner 提供了专门的 RL 训练脚本，您只需指定模型路径、数据集路径和相关训练参数：

```{code-block} bash
:caption: 启动 GRPO 训练

bash examples/v1/run_rl.sh xtuner/examples/v1/config/rl_qwen3_8B_grpo_tiny.py "lmdeploy" <qwen3-8B模型路径> tests/resource/gsm8k_train_example_data.jsonl

```

执行命令后，您将看到类似以下的日志输出，表明 RL 训练已成功启动：

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

训练完成后，工作目录下将保存经过强化学习优化的模型权重。

```{hint}
想进一步了解 RL 训练的详细配置和自定义流程吗？
- [使用 Python 代码自定义 GRPO 训练](../rl/tutorial/rl_grpo_trainer.rst)
- [RL Trainer详解](../api/rl_trainer.rst)
- [RL 训练配置详解](../api/rl_config.rst)
```
