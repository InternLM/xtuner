# Megatron MoE Training Benchmark and Tuning Guide

## Large Model Training Speed Optimization Approach
### First Solve Memory Issues
In large-scale model training, model parallel partitioning must be performed to solve memory issues, but excessive partitioning affects single-card computation. Therefore, our primary goal is to maximize single-card computation as much as possible, mainly by reasonably controlling parallelism and avoiding excessive partitioning.

Before analyzing parallel strategies, let's understand the main components of memory: parameters, gradients, mixed-precision master parameters and gradients, optimizer states, and activation values.

Megatron currently supports various parallel strategies. How should they be chosen?

Zero1 is a must-have option. It effectively reduces the large amount of memory occupied by optimizer states and has low communication overhead with good scalability. In large model training, gradient accumulation is usually performed multiple times (i.e., a larger number of micro batches) before gradient aggregation, so zero1 doesn't introduce significant overhead in such scenarios. However, other strategies have their own advantages and disadvantages.

**TP (Tensor Parallelism):**
- Memory Effect: Can efficiently reduce memory usage. Parameters, gradients, optimizer states, and activation values (when sequence_parallel is enabled) are all proportionally reduced
- Computation: Single-card computation is proportionally reduced
- Computation and Communication Efficiency: Due to large communication volume and smaller matrix partitioning, even with overlap strategies, computation efficiency is reduced
- Recommendation: Only enable TP with sequence_parallel when memory is insufficient, otherwise don't enable it

**PP (Pipeline Parallelism):**
- Memory Effect: Does not reduce activation memory (currently there are ZB-Half, PipeOffload and other technologies exploring PP to further reduce memory, which are worth attention)
- Computation: Single-card computation is proportionally reduced
- Computation and Communication Efficiency: Relatively small communication volume, doesn't reduce matrix multiplication computation scale in kernels, so computation efficiency is high
- Recommendation: PP has good scalability and is recommended to use first. However, PP shouldn't be too large. On one hand, it reduces single-card computation, and on the other hand, when PP is large, micro batch numbers must be increased synchronously (which increases global batch size) to reduce pipeline bubbles.

**EP (Expert Parallelism):**
- Memory Effect: Does not reduce activation memory, nor does it reduce Attention module memory usage
- Computation: Enabling EP does not reduce single-card computation
- [Dual Nature of EP] Computation and Communication Efficiency: The disadvantage is large communication volume and EP load imbalance issues; the advantage is increasing input batch size for individual experts, improving computation density (this is why computation is not reduced in the previous point). To solve EP disadvantages, technologies like DualPipe, load balancing with and without loss have emerged.
- Recommendation: Computation efficiency may increase or decrease, requiring trade-offs based on specific situations.

### Second Goal: Maximize Computation
After solving memory issues, we need to achieve full utilization of computing resources through various means to ensure GPU computing power is maximized. Main optimization methods include:

**Communication Overlap Optimization:**
- **Zero1 Communication Overlap**: Overlap optimizer state communication with backpropagation computation to reduce communication waiting time
- **TP Communication Overlap**: Overlap AllGather, ReduceScatter and other communications with computation operations in tensor parallelism
- **PP Communication Overlap**: Overlap P2P communication with computation in pipeline parallelism to improve pipeline efficiency
- **EP A2A Communication Overlap**: Overlap All-to-All communication with other computation operations in expert parallelism

**Improve Computation Intensity:**
- **Reasonably Control Parallelism**: As mentioned earlier, minimize unnecessary parallel partitioning to increase single-card computation and avoid computation resource waste caused by excessive partitioning
- **Increase Computation Intensity of Computing Units**: Through operator fusion, optimizing matrix multiplication scale and batch size, reduce memory access overhead and improve computing core utilization
- **Control PP Bubble Rate**: Reasonably set micro batch numbers to ensure balanced load across pipeline stages and reduce pipeline bubble time
- **Solve Load Imbalance Problems**:
  - PP Load Imbalance: Ensure relatively balanced computation across pipeline stages
  - EP Load Imbalance: Optimize expert routing strategies to avoid some experts being overloaded while others are idle
- **Parameter Tuning**: Fine-tune MicroBatchSize, SeqLen, Recompute, Offload and other parameters to balance memory usage and computation efficiency for optimal performance
  - Note that using Recompute alone will definitely slow down compared to baseline, but if enabling Recompute can turn off TP or increase MicroBatchSize, it may compensate for recompute overhead by improving computation intensity, thereby improving overall speed

## Megatron Training Qwen3-235B-A22B MoE Model Tuning Practice and Benchmark Results

### Environment Description
Environment: H800 256 cards
Code based on Pai-Megatron-Patch Commit id: [2b201af](https://github.com/alibaba/Pai-Megatron-Patch/commit/2b201af08336dea0403df7c6b497c964cf5a2e75)
Image: dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.04

Configuration same as official Qwen3 example script, other different configurations as follows
```
recompute=select means recomputing layernorm and swiglu in moe, similar to DeepSeekV3's approach, introducing minimal recomputation but saving some activation memory: --recompute-granularity selective --recompute-modules moe_act layernorm
recompute=moe means recomputing layernorm and entire moe module: --recompute-granularity selective --recompute-modules moe layernorm
If TP is enabled, enable --sequcen_parallel
If not specified, default MicroBatchSize mbs=1: --micro-batch-size 1
Enable zero overlap: --overlap-grad-reduce --overlap-param-gather
For speed testing, enable moe forced balancing: --moe-router-force-load-balancing
Disable cuda graph, i.e., remove: --external-cuda-graph --cuda-graph-scope attn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,pinned_use_cuda_host_register:True

Other settings:
--manual-gc --manual-gc-interval 5
--cross-entropy-fusion-impl te
--num-workers 4 --no-mmap-bin-files --no-create-attention-mask-in-dataloader
--moe-permute-fusion
```

### TP vs PP: Prioritize PP, Consider TP Only When Memory is Insufficient

Experimental Results:

Configuration | SeqLen | GlobalBatchSize | GPUNum | TimePerIter (s) | Tokens/GPU/Second
-- | -- | -- | -- | -- | --
tp2, ep8, pp32, vpp3, mbs=1, recompute=select | 4096 | 2048 | 256 | 22.42 | 1461.56
tp4, ep8, pp8, vpp4, mbs=1 | 4096 | 2048 | 256 | 35.481 | 923.53

The key difference between these two configurations is the TP setting. The TP2 configuration is significantly faster than TP4. This confirms the advice above: minimize TP usage.

Can we avoid TP altogether?

In our current scenario of training a 235B model on 256 cards, if we require no recompute (recompute=select is also approximately no recompute), then TP must be enabled, otherwise it will exceed single-card memory.

This is because neither PP nor EP reduce activation memory, only TP with sequcen_parallel can efficiently reduce activation memory. Activation memory is the largest component. Let's look at specific case analysis.

We perform memory case analysis on the above two Qwen3 235B MoE training configurations by theoretically calculating the total memory and proportion of each component on a single GPU.

First, look at it from the memory type dimension, i.e., the proportion of parameters, gradients, mixed-precision master parameters and gradients, optimizer states, and activation values.

```{{figure}} ../../assets/images/benchmark/tp2_tp4_mem_ratio.png

Memory usage with different TP strategies
```

From the above figure, we can see that thanks to relatively large parallelism and zero1 strategy, the proportion of model parameters, gradients, optimizer and other memory is small. In this scenario, activation values are the largest component.
From the analysis in the previous section, we know that neither pipeline parallelism nor expert parallelism can reduce activation memory. Therefore, we can only reduce activation memory through TP (with sequence_parallel), allowing a single H800 to handle 235B model training.

Additionally, let's look at the memory proportion of the above two strategies by Attention and MLP modules, showing that MLP modules account for the majority.

```{{figure}} ../../assets/images/benchmark/tp2_tp4_mem.png

Memory usage with different TP strategies
```


### Trying Different EP Settings

Configuration | SeqLen | GlobalBatchSize | GPUNum | TimePerIter (s) | Tokens/GPU/Second
-- | -- | -- | -- | -- | --
tp2, ep32, pp8, vpp4, recompute=select | 4096 | 2048 | 256 | 35.392 | 925.87
tp2, ep32, deepep, pp8, vpp4, recompute=select | 4096 | 2048 | 256 | 25.73 | 1273.54
tp2, ep8, pp32, vpp3, mbs=1, recompute=select | 4096 | 2048 | 256 | 22.42 | 1461.56

Comparing the first two experiments, both with ep32, we can see that using DeepEP can significantly speed up, reducing EP communication time to compensate for this disadvantage.

Comparing the last two experiments, ep32 pp8 with DeepEP is slower than ep8 pp32, indicating that although increasing ep improves Group GEMM computation intensity, the resulting All-to-All communication greatly affects training efficiency.
Currently, Megatron doesn't have technologies like DualPipe in DeepSeek to hide All-to-All communication, but the Megatron team is developing an Interleaved 1F1B-based A2A overlap technology, refer to their [blog](https://developer.nvidia.com/zh-cn/blog/1f1b-moe-a2a-computing-overlap/).

### Trying Communication Overlap

First, let's look at the experimental results of zero overlap and tp overlap.

Configuration | SeqLen | GlobalBatchSize | GPUNum | TimePerIter (s) | Tokens/GPU/Second
-- | -- | -- | -- | -- | --
tp2, ep8, pp8, recompute=select | 4096 | 2048 | 256 | 27.555 | 1189.17
tp2, ep8, pp8, recompute=select, zero overlap | 4096 | 2048 | 256 | 29.66 | 1104.8
tp2, ep8, pp8, recompute=select, zero/tp overlap | 4096 | 2048 | 256 | 29.276 | 1119.26

The first two experiments show that zero overlap doesn't change much, because the current gradient accumulation times are GlobalBatchSize / DP / MBS = 2048 / 16 / 1 = 128, meaning 128 forward and backward passes before one zero communication, so this isn't the current speed bottleneck.
The last two experiments show that tp overlap also doesn't change much, because although tp overlap overlaps tp communication, the related matrix multiplication scale is smaller, reducing computation efficiency, and the two cancel each other out. If TP increases further, making matrix multiplication scale even smaller, speed will decrease. Similar situations occur in Context Parallel for long sequences.

Let's look at pp overlap experiments.

Configuration | SeqLen | GlobalBatchSize | GPUNum | TimePerIter (s) | Tokens/GPU/Second
-- | -- | -- | -- | -- | --
tp2, ep8, pp32, vpp3, mbs=4, recompute=full | 4096 | 4096 | 256 | 43.495 | 1506.76
tp2, ep8, pp32, mbs=4, recompute=full | 4096 | 8192 | 256 | 92.652 | 1414.68

In Megatron, only enabling virtual pipeline parallel (i.e., interleaved 1F1B) can achieve communication overlap in the pipeline. On one hand, there are implementation reasons, on the other hand, because in ordinary 1F1B, the forward and backward arrangement between different stages is very tight, leaving no space for communication overlap.
Also pay attention to pipeline bubble issues, using the above two configurations as examples.
First configuration:
```
DP=GPUs/PP/TP=4
MBN=256
MBS=4
GBS=4*256*4=4096
vpp=3
Bubble rate=(pp-1)/(MBN*vpp+pp-1)=0.039
```

In the second configuration, since vpp is closed, micro batch number (MBN) must be increased to reduce the bubble rate to a similar level. By the way, in pipeline parallelism, often need to set larger micro batch numbers.
```
DP=GPUs/PP/TP=4
MBN=512
MBS=4
GBS=4*256*2*4=8192
Bubble rate=(pp-1)/(MBN+pp-1)=0.057
```

With similar pipeline bubble rates, we can see that pipeline communication overlap can indeed bring certain speed improvements.

Here, let's talk about PP load balancing issues. The 235B model has 94 transformer layers in total. This design considers that the initial embedding and final loss calculation consume more time, so to prevent the first pipeline stage and last stage from becoming speed bottlenecks, one fewer transformer layer is placed in the first and last stages. This is equivalent to treating embedding and loss calculation as one layer.

### PP Parallelism Setting

How to set PP parallelism numbers? Is smaller always better? Let's fix TP and EP and look at different PP performance.

Configuration | SeqLen | GlobalBatchSize | GPUNum | TimePerIter (s) | Tokens/GPU/Second
-- | -- | -- | -- | -- | --
tp2, ep8, pp8, recompute=select | 4096 | 2048 | 256 | 27.555 | 1189.17
tp2, ep8, pp32, vpp3, recompute=select | 4096 | 2048 | 256 | 22.42 | 1461.56

We can see that the second configuration pp32 speed is not lower than pp8, and due to enabling vpp for pipeline communication overlap, speed is even improved. This demonstrates PP's good scalability.
Assuming activation memory for one micro batch is M, for ordinary 1F1B without vpp, peak activation memory is `pp*M`, and if vpp is enabled, peak activation memory is `(vpp+1)/vpp * pp * M`.
The first pp8 configuration, since memory is basically full, once vpp is enabled, it will OOM due to further increased activation memory.

Therefore, PP number setting has great flexibility and can be considered comprehensively with other settings like EP number, vpp, etc.


### Recomputation

Configuration | SeqLen | GlobalBatchSize | GPUNum | TimePerIter (s) | Tokens/GPU/Second
-- | -- | -- | -- | -- | --
tp2, ep8, pp32, vpp3, recompute=select | 4096 | 2048 | 256 | 22.42 | 1461.56
tp1, ep8, pp32, vpp3, recompute=moe | 4096 | 2048 | 256 | 22.946 | 1428.04
tp1, ep8, pp32, vpp3, mbs=4, recompute=full | 4096 | 4096 | 256 | 42.537 | 1540.68
tp1, ep8, pp32, vpp3, mbs=8, recompute=full | 4096 | 8192 | 256 | 86.186 | 1520.8

In the first configuration, recompute=select means recomputing layernorm and swiglu in moe, i.e., `--recompute-granularity selective --recompute-modules moe_act layernorm`. This is similar to DeepSeekV3's approach, introducing minimal recomputation but saving some activation memory. This configuration's speed can be approximated as no recompute.
In the second configuration, recompute=moe means recomputing layernorm and entire moe module, i.e., `--recompute-granularity selective --recompute-modules moe layernorm`. From previous analysis, we know that activation and MLP (i.e., MoE) memory account for the vast majority, so this setting can save a large amount of activation memory but also introduces more recomputation overhead. Here TP is changed from 2 to 1, i.e., TP is turned off. Compared with no recompute (first configuration), speed is equivalent.
In the third configuration, recompute=full means each transformer layer only retains input hidden states, other activation values are recomputed during backward. This brings greater overhead but also saves more memory. Due to memory savings, larger MicroBatchSize can be used (set to 4 here), thereby improving computation intensity of computing units. Overall speed is significantly improved compared with the first approximately no recompute configuration.
However, as MicroBatchSize continues to increase, speed improvement benefits will peak, which can be verified by looking at the fourth configuration results.

### EP Load Imbalance

All previous experiments for speed testing enabled moe forced balancing `--moe-router-force-load-balancing`. This is because when expert parallelism is enabled and moe routing is unbalanced, OOM situations are likely to occur, and speed will slow down.
How much impact does it have on speed? Here's an experiment with different capacity (`--moe-expert-capacity-factor`).

Configuration | SeqLen | GlobalBatchSize | GPUNum | TimePerIter (s) | Tokens/GPU/Second
-- | -- | -- | -- | -- | --
tp4, ep8, deepep, pp8, vpp4, capacity=1 | 4096 | 2048 | 256 | 50.81 | 644.92
tp4, ep8, deepep, pp8, vpp4, capacity=2 | 4096 | 2048 | 256 | 51.637 | 634.58
tp4, ep8, deepep, pp8, vpp4, capacity=3 | 4096 | 2048 | 256 | 53.734 | 609.82
tp4, ep8, deepep, pp8, vpp4, capacity=5 | 4096 | 2048 | 256 | 66.337 | 493.96

We can see that as capacity increases, speed significantly slows down. This is because expert routing is unbalanced, causing some experts to process too many tokens and become slow nodes, dragging down overall speed.
If capacity is set to larger values, or directly using token dropless strategies, it will be even slower and easily OOM under certain conditions (such as specific data).

### Optimal Strategy Summary and Benchmark Results

After trying multiple strategies, here are the optimal strategies to share.

Configuration | SeqLen | GlobalBatchSize | GPUNum | TimePerIter (s) | Tokens/GPU/Second | TFlops/GPU
-- | -- | -- | -- | -- | -- | --
tp1, ep8, pp32, vpp3, mbs=4, recompute=full | 4096 | 4096 | 256 | 42.537 | 1540.68 | 228.09
tp1, ep8, pp32, vpp3, mbs=8, recompute=full	| 4096 | 8192 | 256 | 86.186| 1520.8 | 225.15
tp2, ep8, pp32, vpp3, mbs=4, recompute=full | 4096 | 4096 | 256 | 43.495 | 1506.76 | 223.08
tp2, ep8, pp32, vpp3, mbs=1, recompute=select | 4096 | 2048 | 256 | 22.42 | 1461.56 | 216.4
tp1, ep8, pp32, vpp3, mbs=1, recompute=moe | 4096 | 2048 | 256 | 22.946 | 1428.04 | 211.52

Current configurations can reach about 1500 TGS, with MFU approximately 228.09/989.4 = 23%.

Recently, NVIDIA also benchmarked the Qwen3 235B model on 256 H100s [benchmark](https://github.com/yanring/Megatron-MoE-ModelZoo/blob/8db3f2c/examples/qwen3/readme.md).
For the main branch code using 3D hybrid parallel strategy, the maximum speed is 200 TFlops, which is similar to our conclusion above.
Additionally, two interesting points can be seen: First, on the main branch code, if using FSDP+recompute strategy instead of 3D parallel strategy, speed can reach 276 TFlops; Second, on another moe_dev branch, testing the unpublished 1F1B overlap strategy, speed can also reach 276 TFlops.

## Hybrid Parallel Best Practice Recommendations

When using hybrid parallelism on Megatron, here are some suggestions:
- Minimize TP usage unless memory pressure is high and TP with sequence_parallel must be enabled.
  - This is essentially because neither PP nor EP reduce activation memory, so from this perspective, recent research on PP technologies to reduce activation values is worth attention, such as ZB-Half, PipeOffload.
- EP has dual nature. On the basis of solving communication overlap and load balancing, enabling large EP has positive benefits
- PP has good scalability and can be set flexibly, but needs to consider Micro Batch Number to avoid large bubble rates
- recompute/offload and other strategies are very worth trying, often bringing faster speed when combined with larger Micro Batch Size