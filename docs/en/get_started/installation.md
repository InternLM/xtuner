# Installation


## Environment Preparation


Ensure that the graphics card driver is correctly installed. For example, on NVIDIA GPU devices, the Driver Version of `nvidia-smi` needs to be greater than `550.127.08`

## XTuner Installation

```{code-block} shell
:caption: Install SFT and Pretrain related dependencies

git clone https://github.com/InternLM/xtuner.git
cd xtuner
pip install -e .
```

Different tasks have different XTuner dependencies. For example, to train gpt-oss models, you need to force install `torch2.8`.

It is recommended to additionally install `GroupedGEMM` for training `MoE` models.

```{code-block} shell
:caption: Install GroupedGEMM

pip install git+https://github.com/InternLM/GroupedGEMM.git@main
```

If you need to train `FP8 MoE` models, in addition to installing the above `GroupedGEMM`, you need to additionally install `AdaptiveGEMM`.
```{code-block} shell
:caption: Install AdaptiveGEMM

pip install git+https://github.com/InternLM/AdaptiveGEMM.git@main
```

```{tip}
:class: margin

For Hopper architecture GPUs, you can additionally install fa3 and enable it through `export XTUNER_USE_FA3=1`
```

In addition, XTuner recommends installing flash-attn, and RL recommends installing flash-attn-3, which can significantly improve training speed. You can refer to the [official documentation](https://github.com/Dao-AILab/flash-attention) for installation.


If you want to experience RL-related features in advance, you need to execute the following command to install RL-related dependencies. In addition, you need to install the inference engine of your choice. Taking LMDeploy as an example, you can refer to the [official documentation](https://github.com/InternLM/lmdeploy/) for installation.

```{code-block} shell
:caption: Install rl related dependencies
pip install -r requirements/rl.txt
# Or install directly
# pip install -e '.[rl]'
```

## XTuner Verification

### LLM Large Model Fine-tuning

Start a simple LLM fine-tuning task on a single card to verify if the installation is successful:

```{note}
:class: margin

Having problems running? Check out the [FAQ](faq)
```


```{code-block} shell
:caption: dense model fine-tuning example
:linenos:

torchrun  xtuner/v1/train/cli/sft.py --model-cfg examples/v1/sft_qwen3_tiny.py --chat_template qwen3 --dataset tests/resource/openai_sft.jsonl
```

After successful execution, the log is as follows

````{toggle}
```shell
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 1/13 data_time: 0.0091 lr: 0.000060 time: 0.6024 text_tokens: 4054.0 total_loss: 12.075, reduced_llm_loss: 12.075 max_memory: 5.36 GB reserved_memory: 7.05 GB grad_norm: 15.847 tgs: 6729.9 e2e_tgs: 6630.0
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 2/13 data_time: 0.0065 lr: 0.000060 time: 0.0515 text_tokens: 3964.0 total_loss: 10.779, reduced_llm_loss: 10.779 max_memory: 7.06 GB reserved_memory: 8.79 GB grad_norm: 9.420 tgs: 76934.8 e2e_tgs: 11936.3
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 3/13 data_time: 0.0064 lr: 0.000059 time: 0.0505 text_tokens: 3898.0 total_loss: 10.176, reduced_llm_loss: 10.176 max_memory: 7.06 GB reserved_memory: 8.79 GB grad_norm: 7.346 tgs: 77123.4 e2e_tgs: 16303.9
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 4/13 data_time: 0.0087 lr: 0.000057 time: 0.0509 text_tokens: 4034.0 total_loss: 9.867, reduced_llm_loss: 9.867 max_memory: 7.06 GB reserved_memory: 8.79 GB grad_norm: 6.393 tgs: 79315.5 e2e_tgs: 20124.4
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 5/13 data_time: 0.0090 lr: 0.000053 time: 0.0523 text_tokens: 3991.0 total_loss: 9.639, reduced_llm_loss: 9.639 max_memory: 7.06 GB reserved_memory: 8.79 GB grad_norm: 6.080 tgs: 76342.2 e2e_tgs: 23295.4
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 6/13 data_time: 0.0063 lr: 0.000047 time: 0.0522 text_tokens: 3998.0 total_loss: 9.455, reduced_llm_loss: 9.455 max_memory: 7.06 GB reserved_memory: 8.79 GB grad_norm: 6.155 tgs: 76605.0 e2e_tgs: 26114.5
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 7/13 data_time: 0.0091 lr: 0.000041 time: 0.0520 text_tokens: 4004.0 total_loss: 9.332, reduced_llm_loss: 9.332 max_memory: 7.06 GB reserved_memory: 8.79 GB grad_norm: 5.986 tgs: 77040.2 e2e_tgs: 28515.1
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 8/13 data_time: 0.0087 lr: 0.000034 time: 0.0499 text_tokens: 3966.0 total_loss: 9.307, reduced_llm_loss: 9.307 max_memory: 7.06 GB reserved_memory: 8.79 GB grad_norm: 5.740 tgs: 79467.9 e2e_tgs: 30659.3
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 9/13 data_time: 0.0062 lr: 0.000027 time: 0.0501 text_tokens: 4083.0 total_loss: 9.091, reduced_llm_loss: 9.091 max_memory: 7.06 GB reserved_memory: 8.79 GB grad_norm: 6.092 tgs: 81479.0 e2e_tgs: 32743.6
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 10/13 data_time: 0.0057 lr: 0.000020 time: 0.0502 text_tokens: 4044.0 total_loss: 9.120, reduced_llm_loss: 9.120 max_memory: 7.06 GB reserved_memory: 8.79 GB grad_norm: 6.090 tgs: 80491.7 e2e_tgs: 34594.4
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 11/13 data_time: 0.0111 lr: 0.000014 time: 0.0548 text_tokens: 4042.0 total_loss: 9.287, reduced_llm_loss: 9.287 max_memory: 7.06 GB reserved_memory: 8.79 GB grad_norm: 5.432 tgs: 73822.0 e2e_tgs: 35971.9
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 12/13 data_time: 0.0051 lr: 0.000008 time: 0.0509 text_tokens: 4010.0 total_loss: 9.007, reduced_llm_loss: 9.007 max_memory: 7.06 GB reserved_memory: 8.79 GB grad_norm: 5.998 tgs: 78743.6 e2e_tgs: 37460.4
[XTuner][RANK 0][2025-09-08 03:04:13][INFO] Step 13/13 data_time: 0.0081 lr: 0.000004 time: 0.0497 text_tokens: 4000.0 total_loss: 9.129, reduced_llm_loss: 9.129 max_memory: 7.06 GB reserved_memory: 8.79 GB grad_norm: 5.567 tgs: 80562.5 e2e_tgs: 38765.3
```
````

The above log shows that only 8G of memory is needed to run. If you want to reduce memory usage further, you can consider modifying the `num_hidden_layers` and `hidden_size` parameters in `examples/v1/sft_qwen3_tiny.py`.

### MLLM Multimodal Large Model Fine-tuning

Start a simple MLLM fine-tuning task on a single card to verify if the installation is successful:

Take Intern-S1 scientific multimodal as an example

```{code-block} shell
:caption: Intern-S1 tiny model fine-tuning example
:linenos:

torchrun xtuner/v1/train/cli/sft.py --config examples/v1/sft_intern_s1_tiny_config.py
```

After successful execution, the log is as follows

````{toggle}
```shell
[XTuner][2025-09-08 03:09:17][INFO] Using toy tokenizer: <xtuner.v1.train.toy_tokenizer.UTF8ByteTokenizer object at 0x7f4c4a256b70>!
[XTuner][2025-09-08 03:09:17][INFO]
============XTuner Training Environment============
XTUNER_DETERMINISTIC: None
XTUNER_FILE_OPEN_CONCURRENCY: None
XTUNER_TOKENIZE_CHUNK_SIZE: None
XTUNER_TOKENIZE_WORKERS: None
XTUNER_ACTIVATION_OFFLOAD: None
XTUNER_USE_FA3: None
XTUNER_GROUP_GEMM: cutlass
XTUNER_DISPATCHER_DEBUG: None
XTUNER_ROUTER_DEBUG: None
==================================================
[XTuner][RANK 0][2025-09-08 03:09:18][WARNING] Model pad_token_id 151645 is different from tokenizer pad_token_id 258. Using tokenizer pad_token_id 258.
[XTuner][RANK 0][2025-09-08 03:09:18][INFO] [mllm_sft_text_example_data.jsonl] Using dynamic image size: True and max_dynamic_patch: 12 and min_dynamic_patch: 1 and use_thumbnail: True data_aug: False for training.
[XTuner][RANK 0][2025-09-08 03:09:18][INFO] Start loading [pure_text]tests/resource/mllm_sft_text_example_data.jsonl with sample_ratio=1.0.
WARNING: input_ids length 4304 exceeds model_max_length 4096. truncated!
WARNING: input_ids length 4639 exceeds model_max_length 4096. truncated!
WARNING: input_ids length 4421 exceeds model_max_length 4096. truncated!
WARNING: input_ids length 5397 exceeds model_max_length 4096. truncated!
[XTuner][RANK 0][2025-09-08 03:09:18][INFO] [Dataset] (Original) pure_text/mllm_sft_text_example_data.jsonl: 200 samples.
[XTuner][RANK 0][2025-09-08 03:09:18][INFO] [mllm_sft_media_example_data.jsonl] Using dynamic image size: True and max_dynamic_patch: 12 and min_dynamic_patch: 1 and use_thumbnail: True data_aug: False for training.
[XTuner][RANK 0][2025-09-08 03:09:18][INFO] Start loading [media]tests/resource/mllm_sft_media_example_data.jsonl with sample_ratio=2.0.
WARNING: input_ids length 4171 exceeds model_max_length 4096. truncated!
WARNING: input_ids length 4188 exceeds model_max_length 4096. truncated!
[XTuner][RANK 0][2025-09-08 03:09:18][INFO] [Dataset] (Original) media/mllm_sft_media_example_data.jsonl: 44 samples.
[XTuner][RANK 0][2025-09-08 03:09:18][INFO] [Dataset] Start packing data of ExpandSoftPackDataset.
[XTuner][RANK 0][2025-09-08 03:09:18][INFO] Using 8 pack workers for packing datasets.
1it [00:00, 294.11it/s]
[XTuner][RANK 0][2025-09-08 03:09:18][INFO] [Dataset] (Original) 244 samples.
[XTuner][RANK 0][2025-09-08 03:09:18][INFO] [Dataset] (Packed) 109 samples.
[FSDP Sharding]:   0%|                                                                                                                                 | 0/8 [00:00<?, ?it/s]/cpfs01/shared/llm_razor/huanghaian/code/refactor_xtuner/xtuner/xtuner/v1/model/utils/checkpointing.py:92: FutureWarning: Please specify CheckpointImpl.NO_REENTRANT as CheckpointImpl.REENTRANT will soon be removed as the default and eventually deprecated.
  return ptd_checkpoint_wrapper(module, *args, **kwargs)
[FSDP Sharding]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 278.75it/s]
[Vision Fully Shard]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 295.44it/s]
[XTuner][RANK 0][2025-09-08 03:09:19][INFO] FSDPInternS1ForConditionalGeneration(
  (vision_tower): FSDPInternS1VisionModel(
    (embeddings): InternVLVisionEmbeddings(
      (patch_embeddings): InternVLVisionPatchEmbeddings(
        (projection): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14))
      )
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (encoder): InternS1VisionEncoder(
      (layer): ModuleList(
        (0-23): 24 x FSDPCheckpointWrapper(
          (_checkpoint_wrapped_module): InternS1VisionLayer(
            (attention): InternS1VisionAttention(
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (projection_layer): Linear(in_features=1024, out_features=1024, bias=True)
              (projection_dropout): Identity()
              (q_norm): Identity()
              (k_norm): Identity()
            )
            (mlp): InternVLVisionMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layernorm_before): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
            (layernorm_after): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (drop_path1): Identity()
            (drop_path2): Identity()
          )
        )
      )
    )
    (layernorm): Identity()
  )
  (multi_modal_projector): FSDPInternS1MultiModalProjector(
    (layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
    (linear_1): Linear(in_features=4096, out_features=1024, bias=True)
    (act): GELUActivation()
    (linear_2): Linear(in_features=1024, out_features=1024, bias=True)
  )
  (language_model): FSDPQwen3Dense(
    (norm): FSDPRMSNorm((1024,), eps=1e-06)
    (lm_head): FSDPLMHead(in_features=1024, out_features=300, bias=False)
    (layers): ModuleDict(
      (0): FSDPCheckpointWrapper(
        (_checkpoint_wrapped_module): DenseDecoderLayer(
          (self_attn): MultiHeadAttention(
            (q_proj): _Linear(in_features=1024, out_features=4096, bias=False)
            (k_proj): _Linear(in_features=1024, out_features=1024, bias=False)
            (v_proj): _Linear(in_features=1024, out_features=1024, bias=False)
            (o_proj): _Linear(in_features=4096, out_features=1024, bias=False)
            (q_norm): RMSNorm((128,), eps=1e-06)
            (k_norm): RMSNorm((128,), eps=1e-06)
          )
          (mlp): DenseMLP(
            (gate_proj): _Linear(in_features=1024, out_features=4096, bias=False)
            (up_proj): _Linear(in_features=1024, out_features=4096, bias=False)
            (down_proj): _Linear(in_features=4096, out_features=1024, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): RMSNorm((1024,), eps=1e-06)
          (post_attention_layernorm): RMSNorm((1024,), eps=1e-06)
        )
      )
      (1): FSDPCheckpointWrapper(
        (_checkpoint_wrapped_module): DenseDecoderLayer(
          (self_attn): MultiHeadAttention(
            (q_proj): _Linear(in_features=1024, out_features=4096, bias=False)
            (k_proj): _Linear(in_features=1024, out_features=1024, bias=False)
            (v_proj): _Linear(in_features=1024, out_features=1024, bias=False)
            (o_proj): _Linear(in_features=4096, out_features=1024, bias=False)
            (q_norm): RMSNorm((128,), eps=1e-06)
            (k_norm): RMSNorm((128,), eps=1e-06)
          )
          (mlp): DenseMLP(
            (gate_proj): _Linear(in_features=1024, out_features=4096, bias=False)
            (up_proj): _Linear(in_features=1024, out_features=4096, bias=False)
            (down_proj): _Linear(in_features=4096, out_features=1024, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): RMSNorm((1024,), eps=1e-06)
          (post_attention_layernorm): RMSNorm((1024,), eps=1e-06)
        )
      )
      ...
    )
    (rotary_emb): RotaryEmbedding()
    (embed_tokens): FSDPEmbedding(300, 1024, padding_idx=258)
  )
)
[XTuner][RANK 0][2025-09-08 03:09:19][INFO] Total trainable parameters: 494.0M, total parameters: 494.0M
[XTuner][RANK 0][2025-09-08 03:09:19][INFO] Untrainable parameters names: []
[XTuner][RANK 0][2025-09-08 03:09:20][INFO] grad_accumulation_steps: 1
[XTuner][RANK 0][2025-09-08 03:09:21][INFO] Step 1/109 data_time: 0.0178 lr: 0.000100 time: 1.3009 text_tokens: 3915.0 total_loss: 5.852, reduced_llm_loss: 5.852 max_memory: 7.49 GB reserved_memory: 7.86 GB grad_norm: 17.179 tgs: 3009.5 e2e_tgs: 2968.7
[XTuner][RANK 0][2025-09-08 03:09:21][INFO] Step 2/109 data_time: 0.0873 lr: 0.000100 time: 0.4088 text_tokens: 3469.0 total_loss: 5.776, reduced_llm_loss: 5.776 max_memory: 8.18 GB reserved_memory: 9.55 GB grad_norm: 20.855 tgs: 8485.0 e2e_tgs: 4067.1
[XTuner][RANK 0][2025-09-08 03:09:22][INFO] Step 3/109 data_time: 0.0741 lr: 0.000100 time: 0.3770 text_tokens: 3901.0 total_loss: 4.781, reduced_llm_loss: 4.781 max_memory: 8.24 GB reserved_memory: 9.77 GB grad_norm: 21.934 tgs: 10348.8 e2e_tgs: 4977.3
[XTuner][RANK 0][2025-09-08 03:09:22][INFO] Step 4/109 data_time: 0.0181 lr: 0.000100 time: 0.3268 text_tokens: 3286.0 total_loss: 5.629, reduced_llm_loss: 5.629 max_memory: 7.66 GB reserved_memory: 9.77 GB grad_norm: 10.874 tgs: 10054.8 e2e_tgs: 5576.8
[XTuner][RANK 0][2025-09-08 03:09:22][INFO] Step 5/109 data_time: 0.0114 lr: 0.000100 time: 0.3039 text_tokens: 3624.0 total_loss: 5.303, reduced_llm_loss: 5.303 max_memory: 7.59 GB reserved_memory: 9.77 GB grad_norm: 8.132 tgs: 11926.8 e2e_tgs: 6212.8
[XTuner][RANK 0][2025-09-08 03:09:23][INFO] Step 6/109 data_time: 0.0734 lr: 0.000100 time: 0.3455 text_tokens: 3602.0 total_loss: 5.339, reduced_llm_loss: 5.339 max_memory: 8.18 GB reserved_memory: 9.77 GB grad_norm: 10.068 tgs: 10424.9 e2e_tgs: 6510.3
[XTuner][RANK 0][2025-09-08 03:09:23][INFO] Step 7/109 data_time: 0.0821 lr: 0.000100 time: 0.3496 text_tokens: 3850.0 total_loss: 4.532, reduced_llm_loss: 4.532 max_memory: 8.18 GB reserved_memory: 9.77 GB grad_norm: 5.207 tgs: 11011.1 e2e_tgs: 6780.2
[XTuner][RANK 0][2025-09-08 03:09:24][INFO] Step 8/109 data_time: 0.0774 lr: 0.000100 time: 0.3452 text_tokens: 3514.0 total_loss: 4.550, reduced_llm_loss: 4.550 max_memory: 8.18 GB reserved_memory: 9.77 GB grad_norm: 4.633 tgs: 10180.4 e2e_tgs: 6933.3
[XTuner][RANK 0][2025-09-08 03:09:24][INFO] Step 9/109 data_time: 0.0132 lr: 0.000100 time: 0.3076 text_tokens: 4036.0 total_loss: 5.225, reduced_llm_loss: 5.225 max_memory: 7.59 GB reserved_memory: 9.77 GB grad_norm: 6.815 tgs: 13122.0 e2e_tgs: 7332.6
[XTuner][RANK 0][2025-09-08 03:09:24][INFO] Step 10/109 data_time: 0.0165 lr: 0.000100 time: 0.3021 text_tokens: 3603.0 total_loss: 4.614, reduced_llm_loss: 4.614 max_memory: 7.66 GB reserved_memory: 9.77 GB grad_norm: 6.599 tgs: 11928.0 e2e_tgs: 7593.1
[XTuner][RANK 0][2025-09-08 03:09:25][INFO] Step 11/109 data_time: 0.0771 lr: 0.000100 time: 0.3457 text_tokens: 3509.0 total_loss: 4.728, reduced_llm_loss: 4.728 max_memory: 8.18 GB reserved_memory: 9.77 GB grad_norm: 7.904 tgs: 10149.9 e2e_tgs: 7648.9
[XTuner][RANK 0][2025-09-08 03:09:25][INFO] Step 12/109 data_time: 0.0128 lr: 0.000100 time: 0.3073 text_tokens: 3624.0 total_loss: 4.677, reduced_llm_loss: 4.677 max_memory: 7.59 GB reserved_memory: 9.77 GB grad_norm: 3.419 tgs: 11792.1 e2e_tgs: 7858.2
WARNING: input_ids length 4171 exceeds model_max_length 4096. truncated!
[XTuner][RANK 0][2025-09-08 03:09:25][INFO] Step 13/109 data_time: 0.0759 lr: 0.000100 time: 0.3467 text_tokens: 4095.0 total_loss: 5.200, reduced_llm_loss: 5.200 max_memory: 8.18 GB reserved_memory: 9.77 GB grad_norm: 7.238 tgs: 11810.1 e2e_tgs: 9000.7
[XTuner][RANK 0][2025-09-08 03:09:26][INFO] Step 14/109 data_time: 0.0135 lr: 0.000100 time: 0.3094 text_tokens: 3813.0 total_loss: 4.531, reduced_llm_loss: 4.531 max_memory: 7.59 GB reserved_memory: 9.77 GB grad_norm: 3.034 tgs: 12323.2 e2e_tgs: 8178.5
[XTuner][RANK 0][2025-09-08 03:09:26][INFO] Step 15/109 data_time: 0.0177 lr: 0.000100 time: 0.3119 text_tokens: 3509.0 total_loss: 4.229, reduced_llm_loss: 4.229 max_memory: 7.66 GB reserved_memory: 9.77 GB grad_norm: 2.872 tgs: 11251.3 e2e_tgs: 8764.3
[XTuner][RANK 0][2025-09-08 03:09:28][INFO] Step 16/109 data_time: 0.0630 lr: 0.000100 time: 0.3474 text_tokens: 3897.0 total_loss: 4.070, reduced_llm_loss: 4.070 max_memory: 8.18 GB reserved_memory: 9.77 GB grad_norm: 6.970 tgs: 11217.2 e2e_tgs: 8798.9
[XTuner][RANK 0][2025-09-08 03:09:28][INFO] Step 17/109 data_time: 0.0187 lr: 0.000100 time: 0.3068 text_tokens: 3827.0 total_loss: 4.054, reduced_llm_loss: 4.054 max_memory: 7.66 GB reserved_memory: 9.77 GB grad_norm: 2.400 tgs: 12473.7 e2e_tgs: 8907.0
[XTuner][RANK 0][2025-09-08 03:09:29][INFO] Step 18/109 data_time: 0.0684 lr: 0.000100 time: 0.3509 text_tokens: 4092.0 total_loss: 3.505, reduced_llm_loss: 3.505 max_memory: 8.24 GB reserved_memory: 9.77 GB grad_norm: 2.953 tgs: 11663.1 e2e_tgs: 8944.9
[XTuner][RANK 0][2025-09-08 03:09:29][INFO] Step 19/109 data_time: 0.0919 lr: 0.000100 time: 0.3813 text_tokens: 3960.0 total_loss: 3.896, reduced_llm_loss: 3.896 max_memory: 8.30 GB reserved_memory: 10.59 GB grad_norm: 3.337 tgs: 10384.5 e2e_tgs: 8916.4
[XTuner][RANK 0][2025-09-08 03:09:30][INFO] Step 20/109 data_time: 0.0234 lr: 0.000100 time: 0.3595 text_tokens: 3967.0 total_loss: 4.578, reduced_llm_loss: 4.578 max_memory: 7.72 GB reserved_memory: 10.59 GB grad_norm: 3.265 tgs: 11035.0 e2e_tgs: 8970.3
WARNING: input_ids length 4171 exceeds model_max_length 4096. truncated!
[XTuner][RANK 0][2025-09-08 03:09:30][INFO] Step 21/109 data_time: 0.0734 lr: 0.000100 time: 0.3467 text_tokens: 4095.0 total_loss: 5.200, reduced_llm_loss: 5.200 max_memory: 8.18 GB reserved_memory: 10.59 GB grad_norm: 7.238 tgs: 11810.1 e2e_tgs: 9000.7
```
````

The above log shows that only 10G of memory is needed to run. If you want to reduce memory usage further, you can consider modifying the `llm_cfg` dictionary related parameters in `examples/v1/sft_intern_s1_tiny_config.py`.

(faq)=
## FAQ

1. ImportError: libGL.so.1: cannot open shared object file: No such file or directory

   Solution:

   ```bash
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```