# Multimodal Large Model Fine-tuning

Before reading this tutorial, please read [Language Model Fine-tuning](./sft.md).

This section will use the [Intern-S1-mini](https://huggingface.co/internlm/Intern-S1-mini) scientific multimodal large model as an example to experience the simplest fine-tuning training startup method. If you want to fine-tune the InternVL model, just replace the model path.

(sft-dataset)=
## Prepare Dataset

Before fine-tuning, you need to prepare the dataset. The multimodal part of XTuner, like SFT, adopts the standard OpenAI format. Just organize the data into `jsonl` format:


```{code-block} json
:caption: jsonl format data example

{"id": 0, "messages": [{"role": "user", "content": "The distance between two stars is 6.52 × 10^5 light years. What is the distance between the two stars in parsecs? (1 parsec = 3.26 light years)\nAnswer Choices: (A) 2 × 10^5 (B) 4 × 10^6 (C) 5 × 10^7 (D) 7 × 10^7 (E) 9 × 10^8"}, {"from": "assistant", "value": "Let's think about the multi-choice question.\n6.52 × 10^5 ly / (3.26 ly/parsec) = 2 x 10^5 persec\nThe answer is A."}]}
{"id": 1, "messages": [{"role": "user", "content": "How many ways can the letters in the word COMMON be arranged?\nAnswer Choices: (A) 6 (B) 30 (C) 90 (D) 120 (E) 180"}, {"from": "assistant", "value": "Let's solve the multi-choice question step by step.\nAccording to the above the # of permutations of 6 letters COMMON out of which 2 O's and 2 M's are identical is 6!2!\u22172!=180\nThe answer is E."}]}
{"id": 2, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {'url':"resource/mscoco_dog_000000319154.jpg","image_wh": [375, 500]}}, {"type": "text", "text": "<IMG_CONTEXT>\nWhat color is the dog in the picture?"}]}, {"role": "assistant", "content": "The dog in the picture is brown."}, {"role": "user", "content": "What type of collar does this dog have?"}, {"role": "assistant", "content": "This dog has a red collar."}, {"role": "user", "content": "What is the position of the dog behind the window?"}, {"role": "assistant", "content": "The dog is sitting behind the window."}, {"role": "user", "content": "What is the dog's expression like when it looks out the window?"}, {"role": "assistant", "content": "When the dog gazes out the window, it looks like it's longing for something. This may suggest that it wants to go outside or interact with someone or something."}, {"role": "user", "content": "Can you describe the dog's fur?"}, {"role": "assistant", "content": "This dog has fluffy, furry brown fur that makes it look both cute and docile."}]}
{"id": 3, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {'url':"resource/mscoco_dog_000000319154.jpg","image_wh": [375, 500]}}, {"type": "text", "text": "<IMG_CONTEXT>\nWhat color is the dog in the picture?"}]}, {"role": "assistant", "content": "The dog in the picture is brown."}]}
```

- The id field is just for convenient data indexing, optional
- The messages field stores conversation content, can include system, user, assistant and other roles
- For image multimodal data, the image_wh field stores the width and height of the image

## Prepare Model

XTuner supports direct fine-tuning of models from Hugging Face. Let's take Intern-S1-mini` as an example, first download the pre-trained model from Hugging Face:


```{code-block} bash
:caption: Download Intern-S1-mini model

# Domestic users can use the huggingface mirror site, set environment variables before executing commands
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download internlm/Intern-S1-mini --local-dir </path/Intern-S1-min>

```

## Start Fine-tuning

After preparing the dataset and model, you can start fine-tuning. XTuner provides a concise command line interface, just specify the model path, dataset path, and training parameters:

```{tip}
:class: margin

OOM? Try `--fsdp-config.cpu-offload`!

```
```{code-block} bash
:caption: Start fine-tuning training
torchrun --nproc-per-node 8  xtuner/v1/train/cli/sft.py  --load-from <model_path>  --dataset <dataset_path>  --total-step 100 --work-dir <target_work_directory>
```

After successfully running the above command, you should see the printed loss is roughly within 1~2, proving that the pre-trained weights are correctly loaded.