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

启动一次简单的微调任务，验证安装是否成功：

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

(额外依赖安装)=
## 额外依赖安装

TODO

