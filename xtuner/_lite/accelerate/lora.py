# Copyright (c) OpenMMLab. All rights reserved.
LORA_TARGET_MAP = {
    "InternLM2ForCausalLM": ["wqkv", "wo", "w1", "w2", "w3"],
    "CLIPVisionModel": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
}
