### Summary

This PR introduces a comprehensive GPU memory calculator for LLM training, which helps users estimate per-GPU memory consumption before actually running training jobs. This tool is essential for capacity planning and optimizing parallel configurations.

### Features

- **Memory Breakdown Analysis**: Calculates detailed memory usage including:
  - Model parameters (embedding, attention, MLP, head)
  - Gradients
  - Optimizer states (Adam momentum & variance)
  - Master parameters and gradients (for mixed-precision training)
  - Intermediate activations

- **Parallel Strategy Support**:
  - Data Parallelism with ZeRO Stage 1(for Megatron)/3 (FSDP for xtuenr)
  - Tensor Parallelism (TP)
  - Pipeline Parallelism (PP) and Virtual Pipeline Parallelism (VPP) for Megatron
   - Pipeline parallelism memory accounts for flying micro-batches during 1F1B schedule
  - Expert Parallelism (EP) for MoE models

- **Model Architecture Support**:
  - Dense models (e.g., Qwen3-8B)
  - MoE models (e.g., Qwen3-235B-A22B, DeepSeek-V3-671B)
  - Multi-Head Attention (MHA)
  - Multi-Latent Attention (MLA) used in DeepSeek

- **Training Optimizations**:
  - Activation checkpointing (recomputation)
  - Flash Attention
  - Chunked loss computation

### Usage

```bash
python xtuner/tools/llm_calc/llm_calculator.py -c <config.yaml> [-o <output.xlsx>]
```

### Example Configurations

| Config File | Model | Scale | TrainFramework |
|------------|-------|-------|-------|
| `xtuner_qwen_dense_8b_1node.yaml` | Qwen3-8B (Dense) | 1 node | xtuner |
| `xtuner_qwen_moe_30b_1node.yaml` | Qwen3-30B-A3B (MoE) | 1 node | xtuner |
| `xtuner_qwen_moe_235b_64node.yaml` | Qwen3-235B-A22B (MoE) | 64 nodes | xtuner |
| `xtuner_deepseekv3_671b_128node.yaml` | DeepSeek-V3-671B (MoE) | 128 nodes | xtuner |
| `megatron_qwen_moe_235b_64node.yaml` | Qwen3-235B-A22B (MoE) | 64 nodes | Megatron |

### Output Example

The tool outputs:
1. Total model parameters breakdown
2. Per-GPU memory consumption for different training phases (forward/backward)
3. Detailed memory breakdown by component (params, grads, optimizer states, activations)
4. Optional Excel export for further analysis
 
```
# python llm_calculator.py -c xtuner_qwen_dense_8b_1node.yaml
----------------------------------------------setting:----------------------------------------------
model:Qwen3-8B zero_stage:3 tp:1 pp:1 dp:8 ep:1 vpp:1 nodes:1 mbs:1 gbs:8 L:32768 recompute:True flash_attn:True capacity:1.0 LN:36
----------------------------------------------------------------------------------------------------
Total_params_num: 7.628208160400391 B, embed_params_num: 0.57958984375 B, attn_params_num: 1.4063873291015625 B, mlp_params_num: 5.0626373291015625 B, head_params_num: 0.5795936584472656 B
Max_mem for microbatch=1: 26.58784294128418 GiB. micro1_forward_last: 26.58784294128418 GiB, micro1_backward_last: 21.386390686035156 GiB
Max_mem for microbatch>1: 30.401947021484375 GiB. micro1_forward_last: 26.58784294128418 GiB, micro1_backward_last: 21.386390686035156 GiB, micro2_forward_last: 30.401947021484375 GiB
---------------------------------------------------------------------memory parts---------------------------------------------------------------------
                 params     grads  opt_states  master_params  master_grads      acts
name
embed          1.159180  1.159180    0.579590       0.289795      0.289795  0.000061
head           1.159187  1.159187    0.579594       0.289797      0.289797  0.625000
perlayer_attn  0.078133  0.078133    0.039066       0.019533      0.019533  1.751953
perlayer_mlp   0.281258  0.281258    0.140629       0.070314      0.070314  2.500000
all_attn       0.000000  0.000000    1.406387       0.703194      0.703194  0.000000
all_mlp        0.000000  0.000000    5.062637       2.531319      2.531319  0.000000
in_layernorm   0.000000  0.000000    0.000000       0.000000      0.000000  0.250000

------------------------------------------------------micro1_forward_last: 26.58784294128418 GiB------------------------------------------------------
  [Static OS&Param] embed, all_attn, all_mlp, head: master_params, opt_states = 11.442312240600586 GiB
  [Allgathered Last Layer] perlayer_attn, perlayer_mlp: params, acts = 4.6113433837890625 GiB
  [Allgathered Head] head: params, acts = 1.7841873168945312 GiB
  [Recompute Checkpoint] in_layernorm.acts * (LN - 1) = 8.75 GiB

-----------------------------------------------------micro1_backward_last: 21.386390686035156 GiB-----------------------------------------------------
  [Static OS&Param] embed, all_attn, all_mlp, head: master_params, opt_states = 11.442312240600586 GiB
  [Static Grad] embed, all_attn, all_mlp, head: master_grads = 3.8141040802001953 GiB
  [Allgathered First Layer] perlayer_attn, perlayer_mlp: params, acts = 4.6113433837890625 GiB
  [Allgathered Embed] embed: params, acts = 1.15924072265625 GiB
  [Previous Layer Grad ReduceScatering] perlayer_attn, perlayer_mlp: grads = 0.3593902587890625 GiB
  [Recompute Checkpoint] None = 0 GiB

-----------------------------------------------------micro2_forward_last: 30.401947021484375 GiB------------------------------------------------------
  [Static OS&Param] embed, all_attn, all_mlp, head: master_params, opt_states = 11.442312240600586 GiB
  [Static Grad] embed, all_attn, all_mlp, head: master_grads = 3.8141040802001953 GiB
  [Allgathered Last Layer] perlayer_attn, perlayer_mlp: params, acts = 4.6113433837890625 GiB
  [Allgathered Head] head: params, acts = 1.7841873168945312 GiB
  [Recompute Checkpoint] in_layernorm.acts * (LN - 1) = 8.75 GiB
```

### Technical Notes for Xtuner FSDP (with recompute) Memory Breakdown of the above example

1. At initilization, each GPU holds sharded master params and optimizer states (usually both in fp32).
```
  [Static OS&Param] embed, all_attn, all_mlp, head: master_params, opt_states = 11.442312240600586 GiB
```

2. At the end of the first micro batch forward: 
2.1 The whole last decoder layer and lm head are gathered in each GPU (in bf16).
```
  [Allgathered Last Layer] perlayer_attn, perlayer_mlp: params, acts = 4.6113433837890625 GiB
  [Allgathered Head] head: params, acts = 1.7841873168945312 GiB
```
2.2 The checkpoint of all decoder layers also exist in each GPU
```
  [Recompute Checkpoint] in_layernorm.acts * (LN - 1) = 8.75 GiB
```

3. At the end of the first micro batch backward
3.1 Each GPU holds master params' graident now
```
  [Static Grad] embed, all_attn, all_mlp, head: master_grads = 3.8141040802001953 GiB
```
3.2 The whole first decoder layer and embedding layer are gathered in each GPU
```
  [Allgathered First Layer] perlayer_attn, perlayer_mlp: params, acts = 4.6113433837890625 GiB
  [Allgathered Embed] embed: params, acts = 1.15924072265625 GiB
```
3.3 FSDP does ReduceScatter of previous layer's gradients at the same time (to overlap computation and communication)
```
  [Previous Layer Grad ReduceScatering] perlayer_attn, perlayer_mlp: grads = 0.3593902587890625 GiB
```
3.4 The checkpoints for recomputing have been released, along with the forward process
```
  [Recompute Checkpoint] None = 0 GiB
```

4. At the end of the second micro batch forward, the only difference with the first micro batch is that each GPU also holds master gradients now.
```
  [Static Grad] embed, all_attn, all_mlp, head: master_grads = 3.8141040802001953 GiB
```

The maximum memory usually occurs in these cases, but maybe different when under different configs such as `chunk_loss_size`.
