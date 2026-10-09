from transformers.models.glm_moe_dsa import GlmMoeDsaForCausalLM


from transformers import AutoProcessor, Glm5NextForConditionalGeneration
import torch

model = Glm5NextForConditionalGeneration.from_pretrained("zai-org/GLM-5.3-Flash")
processor = AutoProcessor.from_pretrained("zai-org/GLM-5.3-Flash")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Describe the image."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
)
inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
generated_ids = model.generate(**inputs, max_new_tokens=64)
