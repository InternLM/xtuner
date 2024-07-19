## Introduction to DPO

### Overview

DPO (Direct Preference Optimization) is a method used in large language model training for directly optimizing human preferences. Unlike traditional reinforcement learning methods, DPO directly uses human preference data to optimize the model, thereby improving the quality of generated content to better align with human preferences. DPO also eliminates the need to train a Reward Model and a Critic Model, avoiding the complexity of reinforcement learning algorithms, reducing training overhead, and enhancing training efficiency.

Many algorithms have made certain improvements to DPO's loss function. In XTuner, besides DPO, we have also implemented loss functions from papers such as [Identity Preference Optimization (IPO)](https://huggingface.co/papers/2310.12036). To use these algorithms, please refer to the [Modify DPO Settings](./modify_settings.md) section. We also provide some [example configurations](https://github.com/InternLM/xtuner/tree/main/xtuner/configs/dpo) for reference.

In addition to DPO, there are alignment algorithms like [ORPO](https://arxiv.org/abs/2403.07691) that do not require a reference model. ORPO uses the concept of odds ratio to optimize the model by penalizing rejected samples during the training process, thereby adapting more effectively to the chosen samples. ORPO eliminates the dependence on a reference model, making the training process more simplified and efficient. The training method for ORPO in XTuner is very similar to DPO, and we provide some [example configurations](https://github.com/InternLM/xtuner/tree/main/xtuner/configs/orpo). Users can refer to the DPO tutorial to modify the configuration.

### Features of DPO Training in XTuner

DPO training in XTuner offers the following significant advantages:

1. **Latest Algorithms**: In addition to supporting standard DPO, XTuner also supports improved DPO algorithms or memory efficient algorithms like ORPO that do not rely on reference models.

2. **Reducing Memory Waste**: Due to the length differences in chosen and rejected data in preference datasets, padding tokens during data concatenation can cause memory waste. In XTuner, by utilizing the variable-length attention feature from Flash Attention2, preference pairs are packed into the same sequence during training, significantly reducing memory waste caused by padding tokens. This not only improves memory efficiency but also allows for training larger models or handling more data under the same hardware conditions.

   ![img](../../zh_cn/reward_model/images/var_len_atten.png)

3. **Efficient Training**:  Leveraging XTuner's QLoRA training capabilities, the reference model can be converted into a policy model with the LoRA adapter removed, eliminating the memory overhead of the reference model weights and significantly reducing DPO training costs.

4. **Long Text Training**: With XTuner's sequence parallel functionality, long text data can be trained efficiently.

### Getting Started

Refer to the [Quick Start Guide](./quick_start.md) to understand the basic concepts. For more information on configuring training parameters, please see the [Modify DPO Settings](./modify_settings.md) section.
