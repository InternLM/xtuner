# 安装


## 环境准备


```{note}
:class: margin

满血版 XTuner 会需要源码编译一些依赖包，详情参考[额外依赖安装](额外依赖安装)
```

确保显卡驱动正确安装即可，例如在 NVIDIA GPU 设备上，`nvidia-smi` 的 Driver Version 需要大于 `550.127.08` 

## XTuner 安装

```shell
git clone https://github.com/InternLM/xtuner.git
cd xtuner
pip install -e .
```


## XTuner 校验

### LLM 大模型微调

单卡启动一次简单的 LLM 微调任务，验证安装是否成功：

```{code-block} shell
:caption: dense 模型微调示例
:linenos:

torchrun  xtuner/v1/train/cli/sft.py --model-cfg examples/v1/sft_qwen3_tiny.py --dataset tests/resource/openai_sft.jsonl
```

运行成功后，日志如下所示

````{toggle}
```shell

[XTuner][2025-08-28 11:30:12][INFO] Using toy tokenizer: <xtuner.v1.train.toy_tokenizer.UTF8ByteTokenizer object at 0x7f428e856ab0>!
[XTuner][RANK 0][2025-08-28 11:30:14][INFO] Start loading [default]/cpfs01/user/yehaochen/codebase/xtuner-init-weights/tests/resource/openai_sft.jsonl with sample_ratio=1.0.
[XTuner][RANK 0][2025-08-28 11:30:14][INFO] [Dataset] (Original) default/openai_sft.jsonl: 128 samples.
[XTuner][RANK 0][2025-08-28 11:30:14][INFO] [Dataset] Start packing data of SoftPackDataset.
[XTuner][RANK 0][2025-08-28 11:30:14][INFO] [Dataset] (Original) 128 samples.
[XTuner][RANK 0][2025-08-28 11:30:14][INFO] [Dataset] (Packed) 32 samples.
[FSDP Sharding]:   0%|                                                                                                                                                                                                                                                                                | 0/3 [00:00<?, ?it/s]
/cpfs01/user/yehaochen/codebase/xtuner-init-weights/xtuner/v1/model/utils/checkpointing.py:92: FutureWarning: Please specify CheckpointImpl.NO_REENTRANT as CheckpointImpl.REENTRANT will soon be removed as the default and eventually deprecated.
  return ptd_checkpoint_wrapper(module, *args, **kwargs)
[FSDP Sharding]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 141.81it/s]
[XTuner][RANK 0][2025-08-28 11:30:15][INFO] FSDPQwen3Dense(
  (norm): FSDPRMSNorm((512,), eps=1e-06)
  (lm_head): FSDPLMHead(in_features=512, out_features=151936, bias=False)
  (layers): ModuleDict(
    (0): FSDPCheckpointWrapper(
      (_checkpoint_wrapped_module): DenseDecoderLayer(
        (self_attn): MultiHeadAttention(
          (q_proj): _Linear(in_features=512, out_features=4096, bias=False)
          (k_proj): _Linear(in_features=512, out_features=1024, bias=False)
          (v_proj): _Linear(in_features=512, out_features=1024, bias=False)
          (o_proj): _Linear(in_features=4096, out_features=512, bias=False)
          (q_norm): RMSNorm((128,), eps=1e-06)
          (k_norm): RMSNorm((128,), eps=1e-06)
        )
        (mlp): DenseMLP(
          (gate_proj): _Linear(in_features=512, out_features=12288, bias=False)
          (up_proj): _Linear(in_features=512, out_features=12288, bias=False)
          (down_proj): _Linear(in_features=12288, out_features=512, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): RMSNorm((512,), eps=1e-06)
        (post_attention_layernorm): RMSNorm((512,), eps=1e-06)
      )
    )
    (1): FSDPCheckpointWrapper(
      (_checkpoint_wrapped_module): DenseDecoderLayer(
        (self_attn): MultiHeadAttention(
          (q_proj): _Linear(in_features=512, out_features=4096, bias=False)
          (k_proj): _Linear(in_features=512, out_features=1024, bias=False)
          (v_proj): _Linear(in_features=512, out_features=1024, bias=False)
          (o_proj): _Linear(in_features=4096, out_features=512, bias=False)
          (q_norm): RMSNorm((128,), eps=1e-06)
          (k_norm): RMSNorm((128,), eps=1e-06)
        )
        (mlp): DenseMLP(
          (gate_proj): _Linear(in_features=512, out_features=12288, bias=False)
          (up_proj): _Linear(in_features=512, out_features=12288, bias=False)
          (down_proj): _Linear(in_features=12288, out_features=512, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): RMSNorm((512,), eps=1e-06)
        (post_attention_layernorm): RMSNorm((512,), eps=1e-06)
      )
    )
    (2): FSDPDenseDecoderLayer(
      (self_attn): MultiHeadAttention(
        (q_proj): _Linear(in_features=512, out_features=4096, bias=False)
        (k_proj): _Linear(in_features=512, out_features=1024, bias=False)
        (v_proj): _Linear(in_features=512, out_features=1024, bias=False)
        (o_proj): _Linear(in_features=4096, out_features=512, bias=False)
        (q_norm): RMSNorm((128,), eps=1e-06)
        (k_norm): RMSNorm((128,), eps=1e-06)
      )
      (mlp): DenseMLP(
        (gate_proj): _Linear(in_features=512, out_features=12288, bias=False)
        (up_proj): _Linear(in_features=512, out_features=12288, bias=False)
        (down_proj): _Linear(in_features=12288, out_features=512, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): RMSNorm((512,), eps=1e-06)
      (post_attention_layernorm): RMSNorm((512,), eps=1e-06)
    )
  )
  (rotary_emb): RotaryEmbedding()
  (embed_tokens): FSDPEmbedding(151936, 512, padding_idx=151645)
)
[XTuner][RANK 0][2025-08-28 11:30:15][INFO] Total trainable parameters: 227.0M, total parameters: 227.0M
[XTuner][RANK 0][2025-08-28 11:30:15][INFO] Trainable parameters names: ['norm.weight', 'lm_head.weight', 'layers.0._checkpoint_wrapped_module.self_attn.q_proj.weight', 'layers.0._checkpoint_wrapped_module.self_attn.k_proj.weight', 'layers.0._checkpoint_wrapped_module.self_attn.v_proj.weight', 'layers.0._checkpoint
_wrapped_module.self_attn.o_proj.weight', 'layers.0._checkpoint_wrapped_module.self_attn.q_norm.weight', 'layers.0._checkpoint_wrapped_module.self_attn.k_norm.weight', 'layers.0._checkpoint_wrapped_module.mlp.gate_proj.weight', 'layers.0._checkpoint_wrapped_module.mlp.up_proj.weight', 'layers.0._checkpoint_wrapped_
module.mlp.down_proj.weight', 'layers.0._checkpoint_wrapped_module.input_layernorm.weight', 'layers.0._checkpoint_wrapped_module.post_attention_layernorm.weight', 'layers.1._checkpoint_wrapped_module.self_attn.q_proj.weight', 'layers.1._checkpoint_wrapped_module.self_attn.k_proj.weight', 'layers.1._checkpoint_wrapp
ed_module.self_attn.v_proj.weight', 'layers.1._checkpoint_wrapped_module.self_attn.o_proj.weight', 'layers.1._checkpoint_wrapped_module.self_attn.q_norm.weight', 'layers.1._checkpoint_wrapped_module.self_attn.k_norm.weight', 'layers.1._checkpoint_wrapped_module.mlp.gate_proj.weight', 'layers.1._checkpoint_wrapped_m
odule.mlp.up_proj.weight', 'layers.1._checkpoint_wrapped_module.mlp.down_proj.weight', 'layers.1._checkpoint_wrapped_module.input_layernorm.weight', 'layers.1._checkpoint_wrapped_module.post_attention_layernorm.weight', 'layers.2.self_attn.q_proj.weight', 'layers.2.self_attn.k_proj.weight', 'layers.2.self_attn.v_pr
oj.weight', 'layers.2.self_attn.o_proj.weight', 'layers.2.self_attn.q_norm.weight', 'layers.2.self_attn.k_norm.weight', 'layers.2.mlp.gate_proj.weight', 'layers.2.mlp.up_proj.weight', 'layers.2.mlp.down_proj.weight', 'layers.2.input_layernorm.weight', 'layers.2.post_attention_layernorm.weight', 'embed_tokens.weight
']
[XTuner][RANK 0][2025-08-28 11:30:15][INFO] grad_accumulation_steps: 1
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 1/32 data_time: 0.0039 lr: 0.000060 time: 0.5577 text_tokens: 1125.0 total_loss: 12.134 reduced_llm_loss: 12.134 max_memory: 4.86 GB reserved_memory: 6.86 GB grad_norm: 16.711 tgs: 2017.2 e2e_tgs: 2003.2 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 2/32 data_time: 0.0058 lr: 0.000060 time: 0.0316 text_tokens: 1593.0 total_loss: 10.822 reduced_llm_loss: 10.822 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 9.256 tgs: 50445.6 e2e_tgs: 4529.2 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 3/32 data_time: 0.0073 lr: 0.000059 time: 0.0311 text_tokens: 2019.0 total_loss: 10.300 reduced_llm_loss: 10.300 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 6.956 tgs: 64928.5 e2e_tgs: 7355.3 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 4/32 data_time: 0.0014 lr: 0.000059 time: 0.0303 text_tokens: 324.0 total_loss: 9.817 reduced_llm_loss: 9.817 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 8.156 tgs: 10682.3 e2e_tgs: 7431.9 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 5/32 data_time: 0.0028 lr: 0.000058 time: 0.0298 text_tokens: 1305.0 total_loss: 9.510 reduced_llm_loss: 9.510 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 7.649 tgs: 43835.2 e2e_tgs: 8855.2 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 6/32 data_time: 0.0029 lr: 0.000057 time: 0.0304 text_tokens: 1587.0 total_loss: 9.247 reduced_llm_loss: 9.247 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 7.024 tgs: 52226.8 e2e_tgs: 10497.9 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 7/32 data_time: 0.0028 lr: 0.000055 time: 0.0304 text_tokens: 1898.0 total_loss: 9.239 reduced_llm_loss: 9.239 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 6.471 tgs: 62376.9 e2e_tgs: 12372.5 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 8/32 data_time: 0.0021 lr: 0.000053 time: 0.0303 text_tokens: 1422.0 total_loss: 9.101 reduced_llm_loss: 9.101 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 6.509 tgs: 46989.6 e2e_tgs: 13518.1 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 9/32 data_time: 0.0028 lr: 0.000051 time: 0.0296 text_tokens: 1573.0 total_loss: 9.067 reduced_llm_loss: 9.067 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 6.159 tgs: 53226.1 e2e_tgs: 14737.7 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 10/32 data_time: 0.0047 lr: 0.000049 time: 0.0298 text_tokens: 1700.0 total_loss: 9.196 reduced_llm_loss: 9.196 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.602 tgs: 57007.4 e2e_tgs: 15958.4 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 11/32 data_time: 0.0045 lr: 0.000047 time: 0.0299 text_tokens: 2034.0 total_loss: 8.874 reduced_llm_loss: 8.874 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.851 tgs: 68068.9 e2e_tgs: 17430.9 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 12/32 data_time: 0.0028 lr: 0.000044 time: 0.0299 text_tokens: 1812.0 total_loss: 8.978 reduced_llm_loss: 8.978 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.725 tgs: 60599.0 e2e_tgs: 18592.5 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 13/32 data_time: 0.0026 lr: 0.000042 time: 0.0296 text_tokens: 1675.0 total_loss: 8.753 reduced_llm_loss: 8.753 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.555 tgs: 56508.4 e2e_tgs: 19543.5 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 14/32 data_time: 0.0043 lr: 0.000039 time: 0.0297 text_tokens: 1784.0 total_loss: 8.946 reduced_llm_loss: 8.946 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.354 tgs: 60082.7 e2e_tgs: 20495.5 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 15/32 data_time: 0.0022 lr: 0.000036 time: 0.0303 text_tokens: 1332.0 total_loss: 8.503 reduced_llm_loss: 8.503 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.604 tgs: 43983.0 e2e_tgs: 20999.8 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 16/32 data_time: 0.0036 lr: 0.000033 time: 0.0301 text_tokens: 2047.0 total_loss: 8.524 reduced_llm_loss: 8.524 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.361 tgs: 68083.0 e2e_tgs: 22074.1 
[XTuner][RANK 0][2025-08-28 11:30:16][INFO] Step 17/32 data_time: 0.0032 lr: 0.000030 time: 0.0297 text_tokens: 1362.0 total_loss: 8.406 reduced_llm_loss: 8.406 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.584 tgs: 45835.7 e2e_tgs: 22512.2 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 18/32 data_time: 0.0031 lr: 0.000028 time: 0.0297 text_tokens: 1340.0 total_loss: 8.350 reduced_llm_loss: 8.350 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.349 tgs: 45093.9 e2e_tgs: 22907.1 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 19/32 data_time: 0.0028 lr: 0.000025 time: 0.0298 text_tokens: 1404.0 total_loss: 8.238 reduced_llm_loss: 8.238 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.252 tgs: 47082.2 e2e_tgs: 23332.1 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 20/32 data_time: 0.0048 lr: 0.000022 time: 0.0304 text_tokens: 1978.0 total_loss: 8.260 reduced_llm_loss: 8.260 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.119 tgs: 65005.5 e2e_tgs: 24124.3 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 21/32 data_time: 0.0027 lr: 0.000019 time: 0.0299 text_tokens: 1853.0 total_loss: 8.397 reduced_llm_loss: 8.397 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 4.986 tgs: 61908.9 e2e_tgs: 24824.9 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 22/32 data_time: 0.0037 lr: 0.000017 time: 0.0301 text_tokens: 1609.0 total_loss: 8.270 reduced_llm_loss: 8.270 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 4.989 tgs: 53521.9 e2e_tgs: 25289.0 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 23/32 data_time: 0.0046 lr: 0.000014 time: 0.0342 text_tokens: 1713.0 total_loss: 8.147 reduced_llm_loss: 8.147 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.074 tgs: 50049.4 e2e_tgs: 25708.9 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 24/32 data_time: 0.0030 lr: 0.000012 time: 0.0301 text_tokens: 1991.0 total_loss: 8.370 reduced_llm_loss: 8.370 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 4.970 tgs: 66158.5 e2e_tgs: 26396.2 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 25/32 data_time: 0.0035 lr: 0.000010 time: 0.0309 text_tokens: 1829.0 total_loss: 8.110 reduced_llm_loss: 8.110 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.230 tgs: 59239.8 e2e_tgs: 26917.6 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 26/32 data_time: 0.0028 lr: 0.000008 time: 0.0314 text_tokens: 1400.0 total_loss: 7.998 reduced_llm_loss: 7.998 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.295 tgs: 44620.9 e2e_tgs: 27155.6 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 27/32 data_time: 0.0016 lr: 0.000006 time: 0.0305 text_tokens: 1547.0 total_loss: 8.113 reduced_llm_loss: 8.113 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.233 tgs: 50725.4 e2e_tgs: 27493.5 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 28/32 data_time: 0.0040 lr: 0.000004 time: 0.0309 text_tokens: 1258.0 total_loss: 8.081 reduced_llm_loss: 8.081 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.153 tgs: 40712.9 e2e_tgs: 27588.2 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 29/32 data_time: 0.0030 lr: 0.000003 time: 0.0303 text_tokens: 1870.0 total_loss: 8.244 reduced_llm_loss: 8.244 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 4.918 tgs: 61775.7 e2e_tgs: 28074.7 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 30/32 data_time: 0.0041 lr: 0.000002 time: 0.0323 text_tokens: 1915.0 total_loss: 8.102 reduced_llm_loss: 8.102 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.026 tgs: 59269.6 e2e_tgs: 28512.4 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 31/32 data_time: 0.0034 lr: 0.000002 time: 0.0293 text_tokens: 1927.0 total_loss: 8.140 reduced_llm_loss: 8.140 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 4.911 tgs: 65717.7 e2e_tgs: 28996.8 
[XTuner][RANK 0][2025-08-28 11:30:17][INFO] Step 32/32 data_time: 0.0038 lr: 0.000001 time: 0.0293 text_tokens: 1971.0 total_loss: 8.318 reduced_llm_loss: 8.318 max_memory: 6.59 GB reserved_memory: 8.02 GB grad_norm: 5.163 tgs: 67381.5 e2e_tgs: 29481.6 
```
````

上述日志显示最大仅需 8G 显存即可运行，如果你还想降低显存占用，可以考虑修改 `examples/v1/sft_qwen3_tiny.py` 中的 `num_hidden_layers` 和 `hidden_size` 参数。

### MLLM 多模态大模型微调

单卡启动一次简单的 MLLM 微调任务，验证安装是否成功：

以 Intern-S1 科学多模态为例

```{code-block} shell
:caption: Intern-S1 tiny 模型微调示例
:linenos:

torchrun xtuner/v1/train/cli/sft.py --trainer-cfg-path examples/v1/sft_intern_s1_tiny_config.py
```

运行成功后，日志如下所示

````{toggle}
```shell
[XTuner][2025-08-29 05:56:24][WARNING] Model pad_token_id 151645 is different from tokenizer pad_token_id 258. Using tokenizer pad_token_id 258.
[XTuner][2025-08-29 05:56:24][INFO] [mllm_sft_text_example_data.jsonl] Using dynamic image size: True and max_dynamic_patch: 12 and min_dynamic_patch: 1 and use_thumbnail: True data_aug: False for training.
[XTuner][2025-08-29 05:56:24][INFO] Start loading [pure_text]tests/resource/mllm_sft_text_example_data.jsonl with sample_ratio=1.0.
[XTuner][2025-08-29 05:56:24][INFO] [Dataset] (Original) pure_text/mllm_sft_text_example_data.jsonl: 200 samples.
[XTuner][2025-08-29 05:56:24][INFO] [mllm_sft_media_example_data.jsonl] Using dynamic image size: True and max_dynamic_patch: 12 and min_dynamic_patch: 1 and use_thumbnail: True data_aug: False for training.
[XTuner][2025-08-29 05:56:24][INFO] Start loading [media]tests/resource/mllm_sft_media_example_data.jsonl with sample_ratio=2.0.
[XTuner][2025-08-29 05:56:24][INFO] [Dataset] (Original) media/mllm_sft_media_example_data.jsonl: 44 samples.
[XTuner][2025-08-29 05:56:24][INFO] [Dataset] Start packing data of SoftPackDataset.
[XTuner][2025-08-29 05:56:24][INFO] [Dataset] (Original) 244 samples.
[XTuner][2025-08-29 05:56:24][INFO] [Dataset] (Packed) 127 samples.
[XTuner][2025-08-29 05:56:25][INFO] FSDPInternS1ForConditionalGeneration(
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
      (2): FSDPCheckpointWrapper(
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
      (3): FSDPCheckpointWrapper(
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
      (4): FSDPCheckpointWrapper(
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
      (5): FSDPCheckpointWrapper(
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
      (6): FSDPCheckpointWrapper(
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
      (7): FSDPDenseDecoderLayer(
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
    (rotary_emb): RotaryEmbedding()
    (embed_tokens): FSDPEmbedding(300, 1024, padding_idx=258)
  )
)
[XTuner][2025-08-29 05:56:25][INFO] Total trainable parameters: 494.0M, total parameters: 494.0M
[XTuner][2025-08-29 05:56:25][INFO] Trainable parameters names: ['vision_tower.embeddings.cls_token', 'vision_tower.embeddings.position_embeddings', 'vision_tower.embeddings.patch_embeddings.projection.weight', 'vision_tower.embeddings.patch_embeddings.projection.bias', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.0._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.1._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.2._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.3._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.4._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.5._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.6._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.7._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.8._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.9._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.10._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.11._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.12._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.13._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.14._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.15._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.16._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.17._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.18._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.19._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.20._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.21._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.22._checkpoint_wrapped_module.layernorm_after.bias', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.lambda_1', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.lambda_2', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.attention.q_proj.weight', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.attention.q_proj.bias', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.attention.k_proj.weight', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.attention.k_proj.bias', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.attention.v_proj.weight', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.attention.v_proj.bias', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.attention.projection_layer.weight', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.attention.projection_layer.bias', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.mlp.fc1.weight', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.mlp.fc1.bias', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.mlp.fc2.weight', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.mlp.fc2.bias', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.layernorm_before.weight', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.layernorm_before.bias', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.layernorm_after.weight', 'vision_tower.encoder.layer.23._checkpoint_wrapped_module.layernorm_after.bias', 'multi_modal_projector.layer_norm.weight', 'multi_modal_projector.layer_norm.bias', 'multi_modal_projector.linear_1.weight', 'multi_modal_projector.linear_1.bias', 'multi_modal_projector.linear_2.weight', 'multi_modal_projector.linear_2.bias', 'language_model.norm.weight', 'language_model.lm_head.weight', 'language_model.layers.0._checkpoint_wrapped_module.self_attn.q_proj.weight', 'language_model.layers.0._checkpoint_wrapped_module.self_attn.k_proj.weight', 'language_model.layers.0._checkpoint_wrapped_module.self_attn.v_proj.weight', 'language_model.layers.0._checkpoint_wrapped_module.self_attn.o_proj.weight', 'language_model.layers.0._checkpoint_wrapped_module.self_attn.q_norm.weight', 'language_model.layers.0._checkpoint_wrapped_module.self_attn.k_norm.weight', 'language_model.layers.0._checkpoint_wrapped_module.mlp.gate_proj.weight', 'language_model.layers.0._checkpoint_wrapped_module.mlp.up_proj.weight', 'language_model.layers.0._checkpoint_wrapped_module.mlp.down_proj.weight', 'language_model.layers.0._checkpoint_wrapped_module.input_layernorm.weight', 'language_model.layers.0._checkpoint_wrapped_module.post_attention_layernorm.weight', 'language_model.layers.1._checkpoint_wrapped_module.self_attn.q_proj.weight', 'language_model.layers.1._checkpoint_wrapped_module.self_attn.k_proj.weight', 'language_model.layers.1._checkpoint_wrapped_module.self_attn.v_proj.weight', 'language_model.layers.1._checkpoint_wrapped_module.self_attn.o_proj.weight', 'language_model.layers.1._checkpoint_wrapped_module.self_attn.q_norm.weight', 'language_model.layers.1._checkpoint_wrapped_module.self_attn.k_norm.weight', 'language_model.layers.1._checkpoint_wrapped_module.mlp.gate_proj.weight', 'language_model.layers.1._checkpoint_wrapped_module.mlp.up_proj.weight', 'language_model.layers.1._checkpoint_wrapped_module.mlp.down_proj.weight', 'language_model.layers.1._checkpoint_wrapped_module.input_layernorm.weight', 'language_model.layers.1._checkpoint_wrapped_module.post_attention_layernorm.weight', 'language_model.layers.2._checkpoint_wrapped_module.self_attn.q_proj.weight', 'language_model.layers.2._checkpoint_wrapped_module.self_attn.k_proj.weight', 'language_model.layers.2._checkpoint_wrapped_module.self_attn.v_proj.weight', 'language_model.layers.2._checkpoint_wrapped_module.self_attn.o_proj.weight', 'language_model.layers.2._checkpoint_wrapped_module.self_attn.q_norm.weight', 'language_model.layers.2._checkpoint_wrapped_module.self_attn.k_norm.weight', 'language_model.layers.2._checkpoint_wrapped_module.mlp.gate_proj.weight', 'language_model.layers.2._checkpoint_wrapped_module.mlp.up_proj.weight', 'language_model.layers.2._checkpoint_wrapped_module.mlp.down_proj.weight', 'language_model.layers.2._checkpoint_wrapped_module.input_layernorm.weight', 'language_model.layers.2._checkpoint_wrapped_module.post_attention_layernorm.weight', 'language_model.layers.3._checkpoint_wrapped_module.self_attn.q_proj.weight', 'language_model.layers.3._checkpoint_wrapped_module.self_attn.k_proj.weight', 'language_model.layers.3._checkpoint_wrapped_module.self_attn.v_proj.weight', 'language_model.layers.3._checkpoint_wrapped_module.self_attn.o_proj.weight', 'language_model.layers.3._checkpoint_wrapped_module.self_attn.q_norm.weight', 'language_model.layers.3._checkpoint_wrapped_module.self_attn.k_norm.weight', 'language_model.layers.3._checkpoint_wrapped_module.mlp.gate_proj.weight', 'language_model.layers.3._checkpoint_wrapped_module.mlp.up_proj.weight', 'language_model.layers.3._checkpoint_wrapped_module.mlp.down_proj.weight', 'language_model.layers.3._checkpoint_wrapped_module.input_layernorm.weight', 'language_model.layers.3._checkpoint_wrapped_module.post_attention_layernorm.weight', 'language_model.layers.4._checkpoint_wrapped_module.self_attn.q_proj.weight', 'language_model.layers.4._checkpoint_wrapped_module.self_attn.k_proj.weight', 'language_model.layers.4._checkpoint_wrapped_module.self_attn.v_proj.weight', 'language_model.layers.4._checkpoint_wrapped_module.self_attn.o_proj.weight', 'language_model.layers.4._checkpoint_wrapped_module.self_attn.q_norm.weight', 'language_model.layers.4._checkpoint_wrapped_module.self_attn.k_norm.weight', 'language_model.layers.4._checkpoint_wrapped_module.mlp.gate_proj.weight', 'language_model.layers.4._checkpoint_wrapped_module.mlp.up_proj.weight', 'language_model.layers.4._checkpoint_wrapped_module.mlp.down_proj.weight', 'language_model.layers.4._checkpoint_wrapped_module.input_layernorm.weight', 'language_model.layers.4._checkpoint_wrapped_module.post_attention_layernorm.weight', 'language_model.layers.5._checkpoint_wrapped_module.self_attn.q_proj.weight', 'language_model.layers.5._checkpoint_wrapped_module.self_attn.k_proj.weight', 'language_model.layers.5._checkpoint_wrapped_module.self_attn.v_proj.weight', 'language_model.layers.5._checkpoint_wrapped_module.self_attn.o_proj.weight', 'language_model.layers.5._checkpoint_wrapped_module.self_attn.q_norm.weight', 'language_model.layers.5._checkpoint_wrapped_module.self_attn.k_norm.weight', 'language_model.layers.5._checkpoint_wrapped_module.mlp.gate_proj.weight', 'language_model.layers.5._checkpoint_wrapped_module.mlp.up_proj.weight', 'language_model.layers.5._checkpoint_wrapped_module.mlp.down_proj.weight', 'language_model.layers.5._checkpoint_wrapped_module.input_layernorm.weight', 'language_model.layers.5._checkpoint_wrapped_module.post_attention_layernorm.weight', 'language_model.layers.6._checkpoint_wrapped_module.self_attn.q_proj.weight', 'language_model.layers.6._checkpoint_wrapped_module.self_attn.k_proj.weight', 'language_model.layers.6._checkpoint_wrapped_module.self_attn.v_proj.weight', 'language_model.layers.6._checkpoint_wrapped_module.self_attn.o_proj.weight', 'language_model.layers.6._checkpoint_wrapped_module.self_attn.q_norm.weight', 'language_model.layers.6._checkpoint_wrapped_module.self_attn.k_norm.weight', 'language_model.layers.6._checkpoint_wrapped_module.mlp.gate_proj.weight', 'language_model.layers.6._checkpoint_wrapped_module.mlp.up_proj.weight', 'language_model.layers.6._checkpoint_wrapped_module.mlp.down_proj.weight', 'language_model.layers.6._checkpoint_wrapped_module.input_layernorm.weight', 'language_model.layers.6._checkpoint_wrapped_module.post_attention_layernorm.weight', 'language_model.layers.7.self_attn.q_proj.weight', 'language_model.layers.7.self_attn.k_proj.weight', 'language_model.layers.7.self_attn.v_proj.weight', 'language_model.layers.7.self_attn.o_proj.weight', 'language_model.layers.7.self_attn.q_norm.weight', 'language_model.layers.7.self_attn.k_norm.weight', 'language_model.layers.7.mlp.gate_proj.weight', 'language_model.layers.7.mlp.up_proj.weight', 'language_model.layers.7.mlp.down_proj.weight', 'language_model.layers.7.input_layernorm.weight', 'language_model.layers.7.post_attention_layernorm.weight', 'language_model.embed_tokens.weight']
[XTuner][2025-08-29 05:56:26][INFO] grad_accumulation_steps: 1
[XTuner][2025-08-29 05:56:28][INFO] Step 1/127 data_time: 0.0270 lr: 0.000100 time: 2.2614 text_tokens: 3577.0 total_loss: 5.662 reduced_llm_loss: 5.662 max_memory: 7.50 GB reserved_memory: 7.85 GB grad_norm: 27.900 tgs: 1581.8 e2e_tgs: 1563.1 
[XTuner][2025-08-29 05:56:28][INFO] Step 2/127 data_time: 0.1008 lr: 0.000100 time: 0.4312 text_tokens: 3984.0 total_loss: 6.317 reduced_llm_loss: 6.317 max_memory: 8.28 GB reserved_memory: 9.72 GB grad_norm: 42.010 tgs: 9239.1 e2e_tgs: 2679.5 
[XTuner][2025-08-29 05:56:29][INFO] Step 3/127 data_time: 0.0809 lr: 0.000100 time: 0.3696 text_tokens: 3509.0 total_loss: 5.407 reduced_llm_loss: 5.407 max_memory: 8.21 GB reserved_memory: 9.72 GB grad_norm: 11.390 tgs: 9494.8 e2e_tgs: 3382.3 
[XTuner][2025-08-29 05:56:29][INFO] Step 4/127 data_time: 0.0204 lr: 0.000100 time: 0.3116 text_tokens: 3230.0 total_loss: 6.112 reduced_llm_loss: 6.112 max_memory: 7.66 GB reserved_memory: 9.72 GB grad_norm: 7.790 tgs: 10367.1 e2e_tgs: 3966.2 
[XTuner][2025-08-29 05:56:29][INFO] Step 5/127 data_time: 0.0162 lr: 0.000100 time: 0.3108 text_tokens: 1721.0 total_loss: 5.872 reduced_llm_loss: 5.872 max_memory: 7.66 GB reserved_memory: 9.72 GB grad_norm: 9.223 tgs: 5537.3 e2e_tgs: 4073.5 
[XTuner][2025-08-29 05:56:30][INFO] Step 6/127 data_time: 0.0802 lr: 0.000100 time: 0.3568 text_tokens: 3895.0 total_loss: 4.927 reduced_llm_loss: 4.927 max_memory: 8.27 GB reserved_memory: 9.72 GB grad_norm: 5.281 tgs: 10917.6 e2e_tgs: 4556.9 
[XTuner][2025-08-29 05:56:30][INFO] Step 7/127 data_time: 0.0728 lr: 0.000100 time: 0.3628 text_tokens: 3613.0 total_loss: 4.952 reduced_llm_loss: 4.952 max_memory: 8.21 GB reserved_memory: 9.72 GB grad_norm: 15.754 tgs: 9958.8 e2e_tgs: 4895.1 
[XTuner][2025-08-29 05:56:31][INFO] Step 8/127 data_time: 0.0725 lr: 0.000100 time: 0.3559 text_tokens: 4033.0 total_loss: 5.269 reduced_llm_loss: 5.269 max_memory: 8.27 GB reserved_memory: 9.72 GB grad_norm: 7.787 tgs: 11332.2 e2e_tgs: 5264.2 
[XTuner][2025-08-29 05:56:31][INFO] Step 9/127 data_time: 0.0714 lr: 0.000100 time: 0.3545 text_tokens: 4055.0 total_loss: 4.391 reduced_llm_loss: 4.391 max_memory: 8.27 GB reserved_memory: 9.72 GB grad_norm: 6.401 tgs: 11437.3 e2e_tgs: 5583.8 
[XTuner][2025-08-29 05:56:32][INFO] Step 10/127 data_time: 0.0685 lr: 0.000100 time: 0.3538 text_tokens: 3803.0 total_loss: 4.103 reduced_llm_loss: 4.103 max_memory: 8.27 GB reserved_memory: 9.72 GB grad_norm: 4.757 tgs: 10747.7 e2e_tgs: 5820.7 
[XTuner][2025-08-29 05:56:32][INFO] Step 11/127 data_time: 0.0676 lr: 0.000100 time: 0.3462 text_tokens: 3513.0 total_loss: 3.633 reduced_llm_loss: 3.633 max_memory: 8.21 GB reserved_memory: 9.72 GB grad_norm: 4.672 tgs: 10148.7 e2e_tgs: 5990.2 
[XTuner][2025-08-29 05:56:32][INFO] Step 12/127 data_time: 0.0077 lr: 0.000100 time: 0.3471 text_tokens: 1479.0 total_loss: 5.262 reduced_llm_loss: 5.262 max_memory: 7.53 GB reserved_memory: 9.72 GB grad_norm: 9.036 tgs: 4260.9 e2e_tgs: 5895.4 
[XTuner][2025-08-29 05:56:33][INFO] Step 13/127 data_time: 0.0106 lr: 0.000100 time: 0.3437 text_tokens: 884.0 total_loss: 3.913 reduced_llm_loss: 3.913 max_memory: 7.60 GB reserved_memory: 9.72 GB grad_norm: 3.103 tgs: 2571.8 e2e_tgs: 5727.8 
[XTuner][2025-08-29 05:56:33][INFO] Step 14/127 data_time: 0.0109 lr: 0.000100 time: 0.3129 text_tokens: 2289.0 total_loss: 4.684 reduced_llm_loss: 4.684 max_memory: 7.60 GB reserved_memory: 9.72 GB grad_norm: 4.295 tgs: 7315.3 e2e_tgs: 5785.0 
[XTuner][2025-08-29 05:56:34][INFO] Step 15/127 data_time: 0.0833 lr: 0.000100 time: 0.3463 text_tokens: 3464.0 total_loss: 5.001 reduced_llm_loss: 5.001 max_memory: 8.21 GB reserved_memory: 9.72 GB grad_norm: 7.237 tgs: 10002.9 e2e_tgs: 5907.6 
[XTuner][2025-08-29 05:56:34][INFO] Step 16/127 data_time: 0.0129 lr: 0.000100 time: 0.3111 text_tokens: 1582.0 total_loss: 4.480 reduced_llm_loss: 4.480 max_memory: 7.60 GB reserved_memory: 9.72 GB grad_norm: 3.907 tgs: 5085.0 e2e_tgs: 5867.2 
[XTuner][2025-08-29 05:56:34][INFO] Step 17/127 data_time: 0.0069 lr: 0.000100 time: 0.3137 text_tokens: 4095.0 total_loss: 4.911 reduced_llm_loss: 4.911 max_memory: 7.53 GB reserved_memory: 9.72 GB grad_norm: 10.390 tgs: 13055.0 e2e_tgs: 6124.0 
[XTuner][2025-08-29 05:56:35][INFO] Step 18/127 data_time: 0.0620 lr: 0.000100 time: 0.3468 text_tokens: 3485.0 total_loss: 3.505 reduced_llm_loss: 3.505 max_memory: 8.21 GB reserved_memory: 9.72 GB grad_norm: 4.942 tgs: 10050.4 e2e_tgs: 6232.5 
[XTuner][2025-08-29 05:56:35][INFO] Step 19/127 data_time: 0.0170 lr: 0.000100 time: 0.3092 text_tokens: 2803.0 total_loss: 4.470 reduced_llm_loss: 4.470 max_memory: 7.66 GB reserved_memory: 9.72 GB grad_norm: 4.743 tgs: 9064.9 e2e_tgs: 6314.6 
[XTuner][2025-08-29 05:56:35][INFO] Step 20/127 data_time: 0.0071 lr: 0.000100 time: 0.3190 text_tokens: 3916.0 total_loss: 4.385 reduced_llm_loss: 4.385 max_memory: 7.53 GB reserved_memory: 9.72 GB grad_norm: 2.890 tgs: 12277.3 e2e_tgs: 6506.2 
[XTuner][2025-08-29 05:56:36][INFO] Step 21/127 data_time: 0.0152 lr: 0.000100 time: 0.3140 text_tokens: 3538.0 total_loss: 4.398 reduced_llm_loss: 4.398 max_memory: 7.66 GB reserved_memory: 9.72 GB grad_norm: 2.436 tgs: 11267.4 e2e_tgs: 6645.4 
[XTuner][2025-08-29 05:56:36][INFO] Step 22/127 data_time: 0.0058 lr: 0.000100 time: 0.3090 text_tokens: 1902.0 total_loss: 4.263 reduced_llm_loss: 4.263 max_memory: 7.53 GB reserved_memory: 9.72 GB grad_norm: 3.086 tgs: 6154.7 e2e_tgs: 6626.7 
[XTuner][2025-08-29 05:56:36][INFO] Step 23/127 data_time: 0.0101 lr: 0.000100 time: 0.3268 text_tokens: 1598.0 total_loss: 4.038 reduced_llm_loss: 4.038 max_memory: 7.60 GB reserved_memory: 9.72 GB grad_norm: 5.155 tgs: 4889.4 e2e_tgs: 6566.8 
[XTuner][2025-08-29 05:56:37][INFO] Step 24/127 data_time: 0.0760 lr: 0.000100 time: 0.3554 text_tokens: 3850.0 total_loss: 4.864 reduced_llm_loss: 4.864 max_memory: 8.21 GB reserved_memory: 9.72 GB grad_norm: 6.481 tgs: 10833.3 e2e_tgs: 6658.1 
[XTuner][2025-08-29 05:56:37][INFO] Step 25/127 data_time: 0.0352 lr: 0.000100 time: 0.3333 text_tokens: 3740.0 total_loss: 4.074 reduced_llm_loss: 4.074 max_memory: 7.81 GB reserved_memory: 9.72 GB grad_norm: 3.227 tgs: 11219.6 e2e_tgs: 6770.1 
```
````

上述日志显示最大仅需 10G 显存即可运行，如果你还想降低显存占用，可以考虑修改 `examples/v1/sft_intern_s1_tiny_config.py` 中的 `llm_cfg` 字典相关参数。


(额外依赖安装)=
## 额外依赖安装

TODO

