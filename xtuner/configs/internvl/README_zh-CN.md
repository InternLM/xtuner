# InterVL 全流程

[English](./README.md) | 简体中文

## InterVL 1.5

### 基本说明

- `./v1_5/` 包含着 InterVL 1.5 训练配置的配置文件
- 支持 InterVL 2B/4B/26B 模型全量/LoRA/QLoRA 微调，综合考虑效率性能，建议你优先选择 4B 模型
- 在训练完成后，可以使用 `./v1_5/convert_to_official.py` 脚本将 XTuner 训练的模型转换为官方格式，从而复用官方所支持的所有工具链
- 目前所有配置都是以 8xA100 80G 显卡为基准，2B/4B 可以使用 ZERO1 训练，26B 模型必须要 ZERO3 运行，并且没有对参数进行过多的调整，你可以按照你自己的需求进行修改
- 目前是以 LLaVA SFT 数据进行验证，无法充分反应微调性能，你可以根据自己的需求进行数据自定义，后续我们会提供一个相对公平的微调数据集

## 数据准备

如果你也想使用 LLaVA SFT 数据集进行训练，请参考[文档](../../../docs/zh_cn/user_guides/dataset_prepare.md#llava-dataset) 准备数据。

对于自定义数据，支持多种 json 和 jsonl 格式，内部数据组织可以参考 LLaVA SFT 格式，且支持数据采样操作。

**(1) 支持多个 json 或者 jsonl 数据**

```text
llava_dataset = dict(
    type=InternVL_V1_5_Dataset,
    model_path=path,
    data_paths=['a.json','b.jsonl','c.json'],
    image_folders=['a',None,'c'],
    template=prompt_template,
    max_length=max_length)
```

**(2) 支持自定义采样**

```text
llava_dataset = dict(
    type=InternVL_V1_5_Dataset,
    model_path=path,
    data_paths=['a.json','b.jsonl','c.json'],
    image_folders=['a',None,'c'],
    repeat_times=[2,0.5,3.5],
    template=prompt_template,
    max_length=max_length)
```

## 训练流程

所提供的配置主要用于基于官方权重继续微调。在准备好数据后，你可以使用以下命令进行训练：

```bash
NPROC_PER_NODE=8 xtuner train internvl_v1_5_phi3_4b_lora_finetune --deepspeed deepspeed_zero1
# NPROC_PER_NODE=8 xtuner train internvl_v1_5_internlm2_26b_lora_finetune.py --deepspeed deepspeed_zero3
```

默认保存在 `./work_dirs/`。

## 模型转换

训练后，我们将获得一组权重即 `./work_dirs/iter_xxx.pth`，为了方便评测和对话，可以将其转换为官方权重。

```bash
python xtuner/configs/internvl/v1_5/convert_to_official.py xtuner/configs/internvl/v1_5/internvl_v1_5_phi3_4b_lora_finetune.py ./work_dirs/iter_xxx.pth ./work_dirs/internvl_v1_5_phi3_4b/
```

此时，会在 `./work_dirs/internvl_v1_5_phi3_4b/` 下生成一组包括配置的完整官方权重，你可以使用官方工具链进行评测和对话。

如果你在使用中碰到任何问题，欢迎联系我们！！！
