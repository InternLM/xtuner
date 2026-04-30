import os
from typing import Any, cast

from xtuner.v1.data_proto.rl_data import RolloutState

from ...data_proto.rl_data import MultimodalInfo
from ..mllm_tokenize_fn.qwen3_vl_tokenize_fn import Qwen3VLTokenizeFnConfig, Qwen3VLTokenizeFunction, QwenVL3DataItem


def remove_consecutive_img_context_tokens(tokens: list[int], img_context_id: int) -> list[int]:
    if not tokens:
        return tokens

    new_tokens = [tokens[0]]
    for i in range(1, len(tokens)):
        if tokens[i] == img_context_id and tokens[i - 1] == img_context_id:
            continue  # 跳过连续的 img_context_id
        else:
            new_tokens.append(tokens[i])
    return new_tokens


def replace_image_context_and_collect_media_data(
    prompt: str | list[dict[str, Any]], media_root: str, replace_image_ctx: bool
) -> tuple:
    """Collect image data from the prompt and extra_info.

    Args:
        prompt (str): The input prompt containing image placeholders.
        media_root (str): The root directory of the media files.
        replace_image_ctx (bool): Whether to replace the image context in the prompt.

    Returns:
        List[dict]: A list of image data dictionaries.
    """
    if not isinstance(prompt, list):
        return [], []

    image_paths = []
    video_paths = []
    for msg in prompt:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                for c in content:
                    if c["type"] in ("image_url", "image"):
                        key = "image_url" if "image_url" in c else "image"
                        image_paths.append(os.path.join(media_root, c[key]["url"]))
                    elif c["type"] in ("video_url", "video"):
                        key = "video_url" if "video_url" in c else "video"
                        video_paths.append(os.path.join(media_root, c[key]["url"]))

    return image_paths, video_paths


class RLQwen3VLTokenizeFunction(Qwen3VLTokenizeFunction):
    def __init__(self, *args, ignore_multimodal_info: bool = False, data_judger_mapping: dict | None = None, **kwargs):
        self.ignore_multimodal_info = ignore_multimodal_info
        self.data_judger_mapping = data_judger_mapping
        super().__init__(*args, **kwargs)

    def __call__(self, item: dict, media_root: str = "", **kwargs) -> RolloutState:
        extra_info = item.get("extra_info", {})
        if isinstance(item["prompt"], dict):
            assert "messages" in item["prompt"], "When prompt is a dict, it must contain 'messages' key"
            assert "tools" in item["prompt"], "When prompt is a dict, it must contain 'tools' key"
            messages = item["prompt"]["messages"]
            tools = item["prompt"]["tools"]
        else:
            messages = item["prompt"]
            tools = None
        system_prompt = getattr(self, "system_prompt", None)
        if system_prompt:
            if messages[0]["role"] == "system":
                messages = messages[1:]
            messages = [{"role": "system", "content": system_prompt}] + messages

        data = super().__call__({"messages": messages, "tools": tools}, media_root=media_root)

        if self.state == "cache":
            return RolloutState(
                message=messages,
                num_tokens=data["num_tokens"],
                proxy_attn_flops=data.get("proxy_attn_flops", float(data["num_tokens"])),
            )
        else:
            data = cast(QwenVL3DataItem, data)
            image_data, _ = replace_image_context_and_collect_media_data(messages, media_root, True)
            if image_data:
                extra_info["image_data"] = image_data

            # 因为 sft tokenizer fn 可能并没有完全和 apply_chat_template 中的 jinja 模块对齐，特别是 system prompt
            # 为了确保一致，必须要通过 tokenizer_fn 得到 prompt_token_ids
            # raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # prompt_token_ids = self.tokenizer(raw_prompt, add_special_tokens=False)["input_ids"]
            prompt_token_ids = remove_consecutive_img_context_tokens(data["input_ids"], self.img_context_token_id)
            raw_prompt = self.tokenizer.decode(prompt_token_ids)  # Just for logging
            extra_info["raw_prompt"] = raw_prompt
            # 训练时的 prompt token ids，包含连续的 img_context_token_id
            extra_info["train_prompt_ids"] = data["input_ids"]

            mm_info = None
            if not self.ignore_multimodal_info:
                mm_info = MultimodalInfo()
                if "pixel_values" in data:
                    mm_info["pixel_values"] = data["pixel_values"].numpy()  # for ray put into shared memory
                if "image_grid_thw" in data:
                    mm_info["image_grid_thw"] = data["image_grid_thw"]

            data_source = item.get("data_source")
            assert data_source is not None, "data_source is required in item"
            extra_info["origin_data_source"] = data_source
            data_judger_mapping = getattr(self, "data_judger_mapping", None)
            if data_judger_mapping is not None:
                mapped_judger_name_and_weight = data_judger_mapping.get(data_source)
            else:
                mapped_judger_name_and_weight = {data_source: 1.0}

            return RolloutState(
                message=messages,
                num_tokens=data["num_tokens"],
                proxy_attn_flops=data.get("proxy_attn_flops", float(data["num_tokens"])),
                prompt_ids=prompt_token_ids,
                position_ids=data["position_ids"],
                data_source=mapped_judger_name_and_weight,
                reward_model=item.get("reward_model", {}),
                mm_info=mm_info,
                extra_fields=extra_info,
            )

    def hash(self) -> str:
        return "RLQwen3VLTokenizeFunction"


class RLQwen3VLTokenizeFnConfig(Qwen3VLTokenizeFnConfig):
    ignore_multimodal_info: bool = False  # eval is True
    data_judger_mapping: dict | None = None  # {origin_data_source: mapped_judger_name_and_weight}

    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str = "", **kwargs
    ) -> RLQwen3VLTokenizeFunction:
        return RLQwen3VLTokenizeFunction(
            tokenizer,
            self.processor_path,
            anno_name,
            chat_template=self.chat_template,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            oss_loader_cfg=self.oss_loader_cfg,
            video_min_total_pixels=self.video_min_total_pixels,
            video_max_total_pixels=self.video_max_total_pixels,
            video_min_frames=self.video_min_frames,
            video_max_frames=self.video_max_frames,
            rand_video_max_frames=self.rand_video_max_frames,
            fps=self.fps,
            enable_3d_rope=self.enable_3d_rope,
            add_vision_id=self.add_vision_id,
            max_length=self.max_length,
            system_message=self.system_message,
            tokenizer_hash=tokenizer_hash,
            ignore_multimodal_info=self.ignore_multimodal_info,
            add_generation_prompt=self.add_generation_prompt,
            enable_thinking=self.enable_thinking,
            data_judger_mapping=self.data_judger_mapping,
        )
