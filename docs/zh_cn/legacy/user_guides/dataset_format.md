# 数据集格式

- [增量预训练数据集格式](#增量预训练数据集格式)
- [单轮对话数据集格式](#单轮对话数据集格式)
- [多轮对话数据集格式](#多轮对话数据集格式)
  - [方法 1](#方法-1)
  - [方法 2](#方法-2)
  - [XTuner 方法介绍](#xtuner-方法介绍)

大语言模型 Supervised Finetune（SFT）旨在通过有监督的微调来提高预训练模型在特定任务上的性能。为支持尽可能多的下游任务，XTuner 支持了增量预训练、单轮对话、多轮对话三种数据集格式。

- 增量预训练数据集用于提升模型在特定领域或任务的能力。
- 单轮对话和多轮对话数据集则经常用于指令微调（instruction tuning）阶段，以提升模型回复特定指令的能力。

在指令微调阶段，我们的目标是训练语言模型根据人类指令给出回答。 **因此，一般只有回答部分（Output）的 loss 会用于梯度回传，而指令部分（System、Input）部分的 loss 则不会用于权重更新。** 基于此，我们在对数据集进行预处理的时候引入了 "system"、"input" 和 "output" 三个字段，"system"、"input" 字段用于保存不需要计算 loss 的文本，例如系统或用户指令，而 "output" 字段则用于保存需要计算 loss 的文本，例如输入指令对应的 GroundTruth 回答。

为了统一增量预训练、单轮对话和多轮对话三种数据集格式，我们将数据集格式设置为以下形式：

```json
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        },
        {
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```

在训练过程中，我们会将一条数据中的多组 "system"、"input" 和 "output" 进行拼接，之后输入模型，并行计算每个位置的 loss ，但只有 "output" 部分对应的 loss 参与梯度回传，如下图所示。

<div  align="center">
<img src="https://github.com/LZHgrla/xtuner/assets/36994684/5ac1ef47-e7e3-43c3-b6b5-5df1aceef970" alt="Image" width="700" />
</div>

其中 \<EOS> token 和 \<BOS> token 用于表示句子或文本的开始和结束。

## 增量预训练数据集格式

由于增量预训练旨在帮助模型学习针对特定下游任务的语言知识和表达能力，因此数据集的全部内容对应的 loss 都应该用于梯度回传。因此，数据集的 "system"、"input" 为空，而 "output" 为一整条语料数据。增量预训练任务对应的数据集格式如下所示：

```json
[{
    "conversation":[
        {
            "system": "",
            "input": "",
            "output": "I am an artificial intelligence (AI) assistant named Puyu. I was created by the Shanghai AI Laboratory and my purpose is to assist users with various tasks through natural language processing technology."
        }
    ]
},
{
    "conversation":[
        {
            "system": "",
            "input": "",
            "output": "I am an artificial intelligence programmed to assist with various types of tasks, including answering questions, providing information, and performing automated processes."
        }
    ]
}]
```

<div  align="center">
<img src="https://github.com/open-mmlab/mmrazor/assets/41630003/f43307b0-09cb-4899-80dd-bfbe2029f550" alt="Image" width="500" />
</div>

## 单轮对话数据集格式

单轮对话数据集往往由一条指令（或问题）及其对应 GroundTruth 回答组成。由于只有回答部分需要对 loss 进行回传，因此数据集的 "system"、"input" 字段为输入指令，"output" 字段为对应回答。单轮对话数据集格式如下所示：

```json
[{
    "conversation":[
        {
            "system": "You are an AI asssistant."
            "input": "Give three tips for staying healthy.",
            "output": "1.Eat a balanced diet. 2. Exercise regularly. 3. Get enough sleep."
        }
    ]
},
{
    "conversation":[
        {
            "system": "You are an AI asssistant."
            "input": "How to study English?",
            "output": "1. Set clear goals. 2. Create a study plan. 3. Build vocabulary. 4. Practice speaking."
        }
    ]
}]
```

<div  align="center">
<img src="https://github.com/LZHgrla/xtuner/assets/36994684/6eed31aa-70e4-47c7-bfdb-20fa7a1312ea" alt="Image" width="700" />
</div>

## 多轮对话数据集格式

多轮对话数据集往往由多轮指令（或问题）+ 对应 GroundTruth 回答组成。假设我们现在有一条多轮对话数据，内容如下。
为方便介绍，对于第 n 轮对话，我们将 User 和 Assistant 对应的输出设为 UserN 和 AssistantN。

```text
System: You are an AI asssistant.
User1：Hello?
Assistant1：Hello! How can I help you?
User2：What's the date today?
Assistant2：Today is Monday, August 14, 2023.
User3：Thank you!
Assistant3：You are welcome.
```

如何使用上述这条多轮对话数据训练大模型？目前有以下两个主流方法。

### 方法 1

System、User1、Assistant1、User2、Assistant2、User3的文本都视为模型的输入部分，将 Assistant3 的文本视为模型的预测部分，只有 Assistant3 部分的 loss 参与权重更新。

<div  align="center">
<img src="https://github.com/LZHgrla/xtuner/assets/36994684/ce869cd5-c1ca-4bc8-9bc3-14f63abb7a5f" alt="Image" width=1100" />
</div>

这种方法的弊端在于没有充分利用多轮对话的训练数据，因为 Assistant1 和 Assistant2 的内容没有参与模型训练，导致训练数据利用率较低。

### 方法 2

将一条多轮对话数据，拆分成多条数据。例如将以上示例拆分成如下三条数据。

<div  align="center">
<img src="https://github.com/LZHgrla/xtuner/assets/36994684/9fd714fc-20bd-4d4c-a4cf-3f95712f1db8" alt="Image" width=1100" />
</div>

相比于方法1，方法2可以充分利用每一轮对话的数据，但需要将一条包含 n 轮对话的数据拆分为 n 条数据，训练效率降低 1/n。

### XTuner 方法介绍

XTuner 训练多轮对话模型时，采取了一种更加充分高效的方法，如下图所示。

<div align="center">
<img src="https://github.com/LZHgrla/xtuner/assets/36994684/ec67b610-a3b2-4fa7-91ad-a9a235fdb820" alt="Image" width=1100" />
</div>

我们将多轮对话进行拼接，之后输入模型，并行计算每个位置的 loss，而只有 Output 部分的 loss 参与回传。因此 XTuner 中多轮对话数据集格式如下所示：

```json
[{
    "conversation":[
        {
            "system": "You are an AI asssistant."
            "input": "Hello?",
            "output": "Hello! How can I help you?"
        },
        {
            "input": "What's the date today?",
            "output": "Today is Monday, August 14, 2023."
        },
        {
            "input": "Thank you!",
            "output": "You are welcome."
        }
    ]
},
{
    "conversation":[
        {
            "system": "You are an AI asssistant."
            "input": "Hello?",
            "output": "Hello! How can I help you?"
        },
        {
            "input": "How's the weather today in Rosso?",
            "output": "The weather in Rosso on Wednesday, August 16th, is going to be cloudy for most of the day, together with moderate rain around noon."
        },
        {
            "input": "Thank you!",
            "output": "You are welcome."
        }
    ]
}]
```

数据集中的 "conversation" 键对应的值是一个列表，用于保存每一轮对话的指令和实际回答（GroundTruth）。为了保持格式统一，增量预训练数据集和单轮对话数据集中的 "conversation" 键也对应一个列表，只不过该列表的长度为 1。而在多轮对话数据集中，"conversation" 列表的长度为 n，以容纳 n 轮的对话内容。
