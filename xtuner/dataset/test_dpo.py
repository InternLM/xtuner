import pytest

from .dpo import DPODataset


@pytest.fixture
def dpo_dataset():
    dpo_dataset_dict = {
        'prompt': [
            'hello',
            'how are you',
            'What is your name?',
        ],
        'chosen': [
            'hi nice to meet you',
            'I am fine',
            'My name is Mary',
        ],
        'rejected': [
            'leave me alone',
            'I am not fine',
            'Whats it to you?',
        ],
    }

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '/public/home/lvshuhang/model_space/workspace/internlm_internlm2-chat-1_8b',
        trust_remote_code=True)
    dataset = DPODataset(dpo_dataset_dict, tokenizer)
    return dataset


@pytest.fixture
def dpo_dataset1():
    """单条样本结构."""
    dpo_dataset_dict = {
        'prompt': 'What is your name?',
        'chosen': 'My name is Mary',
        'rejected': 'Whats it to you?'
    }

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '/public/home/lvshuhang/model_space/workspace/internlm_internlm2-chat-1_8b',
        trust_remote_code=True)
    dataset = DPODataset(dpo_dataset_dict, tokenizer)
    return dataset


# 测试 DPODataset 的长度
def test_dpo_dataset_length(dpo_dataset):
    assert len(dpo_dataset) == 3


# 测试 __getitem__ 方法
def test_dpo_dataset_getitem(dpo_dataset):
    sample = dpo_dataset[0]
    print(f'{sample}')
    # 检查返回字典的键
    assert all(key in sample for key in [
        'input_chosen_ids', 'chosen_labels', 'input_reject_ids',
        'reject_labels'
    ])


def test_dpo_dataset_getitem1(dpo_dataset1):
    sample = dpo_dataset1[0]
    print(f'{sample}')
    # 检查返回字典的键
    assert all(key in sample for key in [
        'input_chosen_ids', 'chosen_labels', 'input_reject_ids',
        'reject_labels'
    ])
