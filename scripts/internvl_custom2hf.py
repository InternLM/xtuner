import os
import sys
import json
import shutil

from collections import OrderedDict
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, AutoModelForImageTextToText


vit_300m_config = {
"architectures": [
    "InternVisionModel"
],
"attention_bias": True,
"attention_dropout": 0.0,
"dropout": 0.0,
"hidden_act": "gelu",
"hidden_dropout_prob": 0.0,
"hidden_size": 1024,
"image_size": [
    448,
    448
],
"initializer_factor": 0.1,
"initializer_range": 1e-10,
"intermediate_size": 4096,
"layer_norm_eps": 1e-06,
"layer_scale_init_value": 0.1,
"model_type": "internvl_vision",
"norm_type": "layer_norm",
"num_attention_heads": 16,
"num_channels": 3,
"num_hidden_layers": 24,
"patch_size": [
    14,
    14
],
"projection_dropout": 0.0,
"torch_dtype": "bfloat16",
"use_absolute_position_embeddings": True,
"use_mask_token": False,
"use_mean_pooling": True,
"use_qk_norm": False
}


vit_6b_config = {
"architectures": [
    "InternVisionModel"
],
"attention_bias": False,
"attention_dropout": 0.0,
"dropout": 0.0,
"hidden_act": "gelu",
"hidden_dropout_prob": 0.0,
"hidden_size": 3200,
"image_size": [
    448,
    448
],
"initializer_factor": 0.1,
"initializer_range": 1e-10,
"intermediate_size": 12800,
"layer_norm_eps": 1e-06,
"layer_scale_init_value": 0.1,
"model_type": "internvl_vision",
"norm_type": "rms_norm",
"num_attention_heads": 25,
"num_channels": 3,
"num_hidden_layers": 45,
"patch_size": [
    14,
    14
],
"projection_dropout": 0.0,
"torch_dtype": "bfloat16",
"use_absolute_position_embeddings": True,
"use_mask_token": False,
"use_mean_pooling": True,
"use_qk_norm": True
}


def convert_chat_config_to_hf(chat_config_path: str,
                              hf_config_path: str,
                              vit_config: dict,
                              image_seq_length: int = 256,
                              image_token_id: int = 151671,
                              use_cache: bool = True,
                              projector_hidden_act: str = "gelu") -> None:
    with open(chat_config_path, "r") as f:
        chat_cfg = json.load(f)

    new_cfg = {
        "architectures": ["InternVLForConditionalGeneration"],
        "downsample_ratio": chat_cfg["downsample_ratio"],
        "image_seq_length": image_seq_length,
        "image_token_id": image_token_id,
        "model_type": "internvl",
        "projector_hidden_act": projector_hidden_act,
        "text_config": chat_cfg["llm_config"],
        "torch_dtype": chat_cfg["torch_dtype"],
        "transformers_version": chat_cfg["transformers_version"],
        "vision_config": vit_config.copy(),
        "vision_feature_layer": -1,
        "vision_feature_select_strategy": "default"
    }
    new_cfg["text_config"]["use_cache"] = use_cache

    with open(hf_config_path, "w") as f:
        json.dump(new_cfg, f, indent=2)

    print(f"转换完成，输出文件: {hf_config_path}")


def convert_keys_to_hf(custom_state_dict):
    new_state_dict = OrderedDict()
    qkv_split_buffer = {}

    for key, value in custom_state_dict.items():
        # === 1. mlp1.* → multi_modal_projector
        if key.startswith("mlp1.0."):
            new_key = "model." + key.replace("mlp1.0.", "multi_modal_projector.layer_norm.")
        elif key.startswith("mlp1.1."):
            new_key = "model." + key.replace("mlp1.1.", "multi_modal_projector.linear_1.")
        elif key.startswith("mlp1.3."):
            new_key = "model." + key.replace("mlp1.3.", "multi_modal_projector.linear_2.")

        # === 2. embeddings ===
        elif key == "vision_model.embeddings.class_embedding":
            new_key = "model.vision_tower.embeddings.cls_token"
        elif key.startswith("vision_model.embeddings.patch_embedding"):
            new_key = "model." + key.replace(
                "vision_model.embeddings.patch_embedding",
                "vision_tower.embeddings.patch_embeddings.projection"
            )
        elif key == "vision_model.embeddings.position_embedding":
            new_key = "model.vision_tower.embeddings.position_embeddings"

        # === 3. encoder ===
        elif key.startswith("vision_model.encoder.layers."):
            parts = key.split(".")
            layer_id = parts[3]
            suffix = ".".join(parts[4:])
            base = f"model.vision_tower.encoder.layer.{layer_id}."

            if suffix.startswith("attn.qkv.weight"):
                qkv_split_buffer[(layer_id, "weight")] = value
                continue
            elif suffix.startswith("attn.qkv.bias"):
                qkv_split_buffer[(layer_id, "bias")] = value
                continue
            elif suffix.startswith("attn.proj."):
                new_key = base + "attention.projection_layer." + suffix.split(".")[-1]
            elif suffix.startswith("norm1."):
                new_key = base + "layernorm_before." + suffix.split(".")[-1]
            elif suffix.startswith("norm2."):
                new_key = base + "layernorm_after." + suffix.split(".")[-1]
            elif suffix == "ls1":
                new_key = base + "lambda_1"
            elif suffix == "ls2":
                new_key = base + "lambda_2"
            else:
                new_key = base + suffix

        # === 4. language_model.model. → language_model.
        elif key == "language_model.lm_head.weight" or key == "language_model.model.lm_head.weight":
            new_key = "lm_head.weight"

        elif key.startswith("language_model.model."):
            new_key = "model." + key.replace("language_model.model.", "language_model.")

        # === 5. already has model. prefix or default
        elif key.startswith("model."):
            new_key = key

        else:
            new_key = "model." + key

        if '.attn.q_norm.' in new_key:
            new_key = new_key.replace('.attn.q_norm.', '.attention.q_norm.')

        if '.attn.k_norm.' in new_key:
            new_key = new_key.replace('.attn.k_norm.', '.attention.k_norm.')

        if '.attn.v_norm.' in new_key:
            new_key = new_key.replace('.attn.v_norm.', '.attention.v_norm.')

        new_state_dict[new_key] = value

    # === 6. 拆分 QKV ===
    for (layer_id, typ), tensor in qkv_split_buffer.items():
        d = tensor.shape[0] // 3
        q, k, v = tensor[:d], tensor[d:2 * d], tensor[2 * d:]
        base = f"model.vision_tower.encoder.layer.{layer_id}.attention."
        if typ == "weight":
            new_state_dict[base + "q_proj.weight"] = q
            new_state_dict[base + "k_proj.weight"] = k
            new_state_dict[base + "v_proj.weight"] = v
        else:
            new_state_dict[base + "q_proj.bias"] = q
            new_state_dict[base + "k_proj.bias"] = k
            new_state_dict[base + "v_proj.bias"] = v

    return new_state_dict


if __name__ == "__main__":
    # mllm_custom_path = sys.argv[1]
    # mllm_save_path = os.path.join(
    #     "/mnt/shared-storage-user/intern7shared/wangweiyun/OpenGVLab-rc1-hf",
    #     f"{os.path.basename(mllm_custom_path)}-HF",
    # )
    
    mllm_custom_path = "/mnt/shared-storage-user/intern7shared/internvl_a4s/xpuyu/work_dir/InternVL3-8B-Qwen3-cpt-science-data-slow-tokenize-data-0628-tokenizer-rjob-h200/20250703154422/hf-30000"
    mllm_save_path = "/mnt/shared-storage-user/intern7shared/wangweiyun/OpenGVLab-rc1-hf/InternVL3_5-8B-CPT-HF"

    print(f"{mllm_custom_path=}")
    print(f"{mllm_save_path=}")

    os.makedirs(mllm_save_path, exist_ok=True)

    # 1. 将自定义模型配置转换为 HF 格式并保存
    chat_config_path = os.path.join(mllm_custom_path, "config.json")
    hf_config_path = os.path.join(mllm_save_path, "config.json")

    if '38B' in mllm_custom_path or '235B' in mllm_custom_path or '241B' in mllm_custom_path:
        vit_config = vit_6b_config
        print('Use ViT-6B')
    else:
        vit_config = vit_300m_config
        print('Use ViT-300M')

    convert_chat_config_to_hf(chat_config_path, hf_config_path, vit_config)

    config = AutoConfig.from_pretrained(mllm_save_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)

    print(f"模型已加载到 GPU，并使用转换后的 HF config: {hf_config_path}")

    # 加载 HF safetensors 权重
    checkpoint_paths = [os.path.join(mllm_custom_path, f) for f in os.listdir(mllm_custom_path) if f.endswith('.safetensors')]
    print(f"\n🔍 Found checkpoint files: {checkpoint_paths}")

    model_state_dict_hf = {}
    for checkpoint_path in checkpoint_paths:
        with safe_open(checkpoint_path, framework="pt") as f:
            for k in f.keys():
                model_state_dict_hf[k] = f.get_tensor(k)

    # 转换为旧 key 命名风格
    model_state_dict = convert_keys_to_hf(model_state_dict_hf)

    # 加载权重
    # missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=True)
    print(f"\n❌ Missing keys: {missing_keys}")
    print(f"⚠️ Unexpected keys: {unexpected_keys}")

    model.save_pretrained(mllm_save_path)

    tokenizer = AutoTokenizer.from_pretrained(
        mllm_custom_path,
        trust_remote_code=True,
        extra_special_tokens={
            "context_image_token": "<IMG_CONTEXT>",
            "start_image_token": "<img>",
            "end_image_token": "</img>",
            "video_token": "<video>",
        },
    )
    tokenizer.add_tokens(['<video>'], special_tokens=True)

    if 'gpt' in mllm_custom_path.lower():
        pass
    else:
        tokenizer.chat_template="""{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'}}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{ '<IMG_CONTEXT>\n' }}{% elif content['type'] == 'video' %}{{ '<video>\n' }}{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{'<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}"""

    tokenizer.model_max_length = model.config.text_config.max_position_embeddings
    tokenizer.save_pretrained(mllm_save_path)

    for file_name in ["processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json"]:
            src_file = os.path.join(mllm_custom_path, file_name)
            if os.path.exists(src_file):
                shutil.copy(src_file, mllm_save_path)
                print(f"✅ 复制文件 {file_name} 到 {mllm_save_path}")
