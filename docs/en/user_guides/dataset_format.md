# Dataset Format

- [Incremental Pre-training Dataset Format](#incremental-pre-training-dataset-format)
- [Single-turn Dialogue Dataset Format](#single-turn-dialogue-dataset-format)
- [Multi-turn Dialogue Dataset Format](#multi-turn-dialogue-dataset-format)
  - [Method 1](#method-1)
  - [Method 2](#method-2)
  - [Method in XTuner](#method-in-xtuner)

The Supervised Finetune (SFT) of large language models aims to improve the performance of pre-trained models on specific tasks through supervised fine-tuning. To support as many downstream tasks as possible, XTuner supports three dataset formats: incremental pre-training, single-turn dialogue, and multi-turn dialogue.

- The incremental pre-training dataset is used to enhance the model's capabilities in a specific domain or task.
- Single-turn and multi-turn dialogue datasets are often used in the instruction tuning stage to enhance the model's ability to respond to specific instructions.

In the instruction tuning phase, our goal is to train the language model to answer based on human instructions. **Therefore, generally only the loss of the response part (Output) is used for gradient backpropagation, while the loss of the instruction part (System, Input) is not used for weight updates.** Based on this, we introduce "system", "input" and "output" fields when preprocessing the dataset. The "system", "input" fields are used to save fields that do not need to compute loss, such as system and user instructions, whereas the "output" field is used to save fields that do need to compute loss, such as the GroundTruth answers corresponding to input instructions.

To unify the incremental pre-training, single-turn dialogue, and multi-turn dialogue dataset formats, we set the dataset format to the following form:

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

Throughout the training phase, we amalgamate several "system", "input" and "output" pairs from a single data instance, which we then feed into the model. Loss is computed concurrently at each position, yet only the loss associated with the "output" component participates in the gradient backpropagation process. This process is elucidated in the figure below.

<div  align="center">
<img src="https://github.com/LZHgrla/xtuner/assets/36994684/5ac1ef47-e7e3-43c3-b6b5-5df1aceef970" alt="Image" width="700" />
</div>

Note that the <EOS> token and <BOS> token are used to indicate the start and end of a sentence or text.

## Incremental Pre-training Dataset Format

As incremental pre-training is intended to help the model learn language knowledge and expressive abilities tailored for specific downstream tasks, the loss corresponding to the entire content of the dataset should be used for gradient backpropagation. Therefore, the "system" and "input" of the dataset are left empty, while the "output" consists of an entire piece of corpus data. The dataset format corresponding to the incremental pre-training task is shown as follows:

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

## Single-turn Dialogue Dataset Format

The single-turn dialogue dataset typically consists of a single instruction (or question) and its corresponding GroundTruth answer. Since only the answer part should be used for gradient backpropagation, the "system" and "input" fields of the dataset are the input instruction, and the "output" field is the corresponding answer. The format of the single-turn dialogue dataset is shown as follows:

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

## Multi-turn Dialogue Dataset Format

The multi-turn dialogue dataset typically consists of multiple rounds of instructions (or questions) and their corresponding GroundTruth answers. Suppose we have a piece of multi-turn dialogue data. For ease of introduction, for the nth round of dialogue, we set the output corresponding to User and Assistant as UserN and AssistantN.

```text
System: You are an AI asssistant.
User1：Hello?
Assistant1：Hello! How can I help you?
User2：What's the date today?
Assistant2：Today is Monday, August 14, 2023.
User3：Thank you!
Assistant3：You are welcome.
```

How can we use the above multi-turn dialogue data to train large models? Currently, there are two mainstream methods.

### Method 1

The text of System, User1, Assistant1, User2, Assistant2, and User3 is all considered as the input part of the model, while the text of Assistant3 is viewed as the prediction part of the model. Only the loss from the Assistant3 part is involved in the weight update.

<div  align="center">
<img src="https://github.com/LZHgrla/xtuner/assets/36994684/ce869cd5-c1ca-4bc8-9bc3-14f63abb7a5f" alt="Image" width=1100" />
</div>

The downside of this method is that it does not fully utilize the multi-turn dialogue training data because the content of Assistant1 and Assistant2 does not participate in model training, leading to a low utilization rate of training data.

### Method 2

Split a piece of multi-turn dialogue data into multiple pieces of data. For example, the above instance can be split into the following three pieces of data.

<div  align="center">
<img src="https://github.com/LZHgrla/xtuner/assets/36994684/9fd714fc-20bd-4d4c-a4cf-3f95712f1db8" alt="Image" width=1100" />
</div>

Compared to Method 1, Method 2 can fully utilize the data from each round of dialogue, but it requires splitting one piece of data containing n rounds of dialogue into n pieces of data, which reduces the training efficiency by 1/n.

### Method in XTuner

When XTuner trains multi-turn dialogue models, it adopts a more comprehensive and efficient method, as shown in the figure below.

<div align="center">
<img src="https://github.com/LZHgrla/xtuner/assets/36994684/ec67b610-a3b2-4fa7-91ad-a9a235fdb820" alt="Image" width=1100" />
</div>

We concatenate multi-turn dialogues, then input them into the model. The loss at each position is computed in parallel, but only the loss from the Output part participates in backpropagation. Therefore, the format of the multi-turn dialogue dataset in XTuner is shown as follows:

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

The value corresponding to the "conversation" key in the dataset is a list used to save the instructions and actual answers (GroundTruth) for each round of dialogue. To maintain uniformity in the format, the value corresponding to the "conversation" key in both incremental pre-training datasets and single-turn dialogue datasets is also a list, albeit with a length of 1. In multi-turn dialogue datasets, the length of the "conversation" list is n to accommodate n rounds of dialogue content.
