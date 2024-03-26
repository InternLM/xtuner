# 速度基准

我们在训练速度方面与 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 进行了对比。对比所使用的 LLaMA-Factory commit id 为 [8e04794](https://github.com/hiyouga/LLaMA-Factory/tree/8e04794b2da067a4123b9d7091a54c5647f44244)。使用 [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) 作为训练数据集测试速度。

## 硬件

- NVIDIA A100-SXM4-80GB GPUs
- Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz

## 软件环境

- Python 3.10
- PyTorch 1.13
- CUDA 11.7
- CUDNN 8.5
- NCCL 2.14.3

## 速度

![image](https://github.com/InternLM/xtuner/assets/41630003/4e93aa63-19ad-46a0-a159-be229f74e624)

![image](https://github.com/InternLM/xtuner/assets/41630003/0ef1ec97-7a03-4109-9faa-aa5196d370af)

![image](https://github.com/InternLM/xtuner/assets/41630003/dc820fd7-30c4-40ec-a4fc-3b0aac71ab25)

|        |   模型    | GPU 数量 | 上下文长度 | 速度 (tokens per second) |                                                               训练 config                                                                |
| :----: | :-------: | :------: | :--------: | :----------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
| XTuner | Llama2-7B |    8     |     8k     |        **3593.0**        |   [llama2_7b_full_alpaca_enzh_8k_sp1.py](../../../xtuner/configs/llama_speed_benchmark/llama2_7b/llama2_7b_full_alpaca_enzh_8k_sp1.py)   |
| XTuner | Llama2-7B |    8     |    32k     |        **3864.2**        |  [llama2_7b_full_alpaca_enzh_32k_sp1.py](../../../xtuner/configs/llama_speed_benchmark/llama2_7b/llama2_7b_full_alpaca_enzh_32k_sp1.py)  |
| XTuner | Llama2-7B |    8     |    128k    |        **3108.9**        | [llama2_7b_full_alpaca_enzh_128k_sp8.py](../../../xtuner/configs/llama_speed_benchmark/llama2_7b/llama2_7b_full_alpaca_enzh_128k_sp8.py) |
| XTuner | Llama2-7B |    8     |    256k    |        **2250.4**        | [llama2_7b_full_alpaca_enzh_256k_sp8.py](../../../xtuner/configs/llama_speed_benchmark/llama2_7b/llama2_7b_full_alpaca_enzh_256k_sp8.py) |
| XTuner | Llama2-7B |    32    |     1M     |        **714.5**         |  [llama2_7b_full_alpaca_enzh_1M_sp16.py](../../../xtuner/configs/llama_speed_benchmark/llama2_7b/llama2_7b_full_alpaca_enzh_1M_sp16.py)  |

|        |    模型     | GPU 数量 | 上下文长度 | 速度 (tokens per second) |                                                                训练 config                                                                |
| :----: | :---------: | :------: | :--------: | :----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------: |
| XTuner | Yi-34B-200K |    32    |     8k     |        **472.7**         |   [yi_34b_200k_full_alpaca_enzh_8k_sp1.py](../../../xtuner/configs/llama_speed_benchmark/yi_34b/yi_34b_200k_full_alpaca_enzh_8k_sp1.py)   |
| XTuner | Yi-34B-200K |    32    |    32k     |        **555.4**         |  [yi_34b_200k_full_alpaca_enzh_32k_sp2.py](../../../xtuner/configs/llama_speed_benchmark/yi_34b/yi_34b_200k_full_alpaca_enzh_32k_sp2.py)  |
| XTuner | Yi-34B-200K |    32    |    128k    |        **625.8**         | [yi_34b_200k_full_alpaca_enzh_128k_sp8.py](../../../xtuner/configs/llama_speed_benchmark/yi_34b/yi_34b_200k_full_alpaca_enzh_128k_sp8.py) |
| XTuner | Yi-34B-200K |    32    |    256k    |        **357.4**         | [yi_34b_200k_full_alpaca_enzh_256k_sp8.py](../../../xtuner/configs/llama_speed_benchmark/yi_34b/yi_34b_200k_full_alpaca_enzh_256k_sp8.py) |

|        |    模型    | GPU 数量 | 上下文长度 | 速度 (tokens per second) |                                                                  训练 config                                                                  |
| :----: | :--------: | :------: | :--------: | :----------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------: |
| XTuner | Llama2-70B |    32    |     8k     |        **216.3**         |    [llama2_70b_full_alpaca_enzh_8k_sp1.py](../../../xtuner/configs/llama_speed_benchmark/llama2_70b/llama2_70b_full_alpaca_enzh_8k_sp1.py)    |
| XTuner | Llama2-70B |    32    |    32k     |        **227.6**         |   [llama2_70b_full_alpaca_enzh_32k_sp4.py](../../../xtuner/configs/llama_speed_benchmark/llama2_70b/llama2_70b_full_alpaca_enzh_32k_sp4.py)   |
| XTuner | Llama2-70B |    32    |    128k    |        **175.6**         |  [llama2_70b_full_alpaca_enzh_128k_sp8.py](../../../xtuner/configs/llama_speed_benchmark/llama2_70b/llama2_70b_full_alpaca_enzh_128k_sp8.py)  |
| XTuner | Llama2-70B |    32    |    256k    |        **108.8**         | [llama2_70b_full_alpaca_enzh_256k_sp16.py](../../../xtuner/configs/llama_speed_benchmark/llama2_70b/llama2_70b_full_alpaca_enzh_256k_sp16.py) |

注：所有实验都会将 Alpaca 数据集拼接为最大长度。由于 Alpaca 数据集所含 token 数较少，无法拼接成超长序列（如 1M 长度），因此当序列长度较长时，会对 XTuner 代码进行如下修改：

```diff
# xtuner/dataset/huggingface.py
def build_origin_dataset(dataset, split):
    ...
+   # 6 times larger dataset (for speed testing purposes only)
+   dataset = concatenate_datasets([dataset for _ in range(6)])
    return dataset

def pack_dataset(dataset, max_length, use_varlen_attn, shuffle_before_pack,
                 map_num_proc):
    dataset = dataset.map(
        Packer(max_length, use_varlen_attn=use_varlen_attn),
        batched=True,
-       num_proc=map_num_proc
+       batch_size=25000,
+       num_proc=1
    )
    return dataset
```

由于 Alpaca 数据量较小，因此做了第一处修改将数据集大小扩大了 6 倍，以保证拥有足够的训练 iter 数（保证速度测试的稳定性）。另外，由于 Alpaca 数据集每条数据的长度较短，因此在数据拼接的时候做了第二处修改以保证拥有足够多的数据，足以拼接为 `max_length` 最大长度。
