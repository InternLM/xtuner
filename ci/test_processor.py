import torch

from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from transformers import AutoTokenizer, AutoProcessor

QWEN3_VL_PATH = 'xxxx'


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_VL_PATH, trust_remote_code=True)
    tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_VL_PATH, add_vision_id=True).build(tokenizer)
    processor = AutoProcessor.from_pretrained(QWEN3_VL_PATH, trust_remote_code=True)

    messages_inference = [
        {
            "role": "user",
            "content": [
                {"type": "time_series",
                 "data": f"{QWEN3_VL_PATH}/0092638_seism.npy",
                 "sampling_rate": 100},
                {"type": "text",
                 "text": "Please determine whether an Earthquake event has occurred in the provided time-series data. If so, please specify the starting time point indices of the P-wave and S-wave in the event."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Yes, an Earthquake event has occurred in the provided time-series data."}]
        }
    ]

    messages_train = [
        {
            "role": "user",
            "content": [
                {"type": "time_series_url",
                 "time_series_url": {
                     'url': f"{QWEN3_VL_PATH}/0092638_seism.npy",
                     "sampling_rate": 100},
                 },
                {"type": "text",
                 "text": "<TS_CONTEXT>Please determine whether an Earthquake event has occurred in the provided time-series data. If so, please specify the starting time point indices of the P-wave and S-wave in the event."},
            ],
        },
        {
            "role": "assistant",
            "content": "Yes, an Earthquake event has occurred in the provided time-series data."
        }
    ]

    time_series_inputs = processor.time_series_preprocessor(messages_inference)
    multimodal_inputs = processor.apply_chat_template(messages_inference, add_generation_prompt=False, tokenize=True,
                                                      return_dict=True, return_tensors="pt", **time_series_inputs)

    output = tokenize_fn({"messages": messages_train})

    assert torch.allclose(multimodal_inputs['input_ids'], torch.tensor(output['input_ids']).reshape(1, -1))
    assert torch.allclose(multimodal_inputs['ts_lens'], torch.tensor(output['ts_len']).reshape(1, -1))
    assert torch.allclose(multimodal_inputs['ts_sr'], torch.tensor(output['ts_sr']).reshape(1, -1))
    assert torch.allclose(multimodal_inputs['ts_values'], torch.tensor(output['time_series_signals']).reshape(1, -1, 3))
