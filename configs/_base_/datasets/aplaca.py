from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from mmchat.datasets import process_hf_dataset

_alpaca = dict(
    type=process_hf_dataset,
    dataset=dict(
        type=load_dataset,
        path='tatsu-lab/alpaca',
    ),
    # map_fn = extract_alpaca_dataset,
    prompt_input_format=(
        'Below is an instruction that describes a task, '
        'paired with an input that provides further context. '
        'Write a response that appropriately completes the request.\n\n'
        '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n'
        '### Response: '),
    prompt_no_input_format=(
        'Below is an instruction that describes a task. '
        'Write a response that appropriately completes the request.\n\n'
        '### Instruction:\n{instruction}\n\n### Response: '),
    remove_columns=['instruction'],
)

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=_alpaca,
    sampler=dict(type=DefaultSampler, shuffle=True))
