from typing import cast

from xtuner.v1.data_proto.rl_data import RolloutState

from ...data_proto.rl_data import MultimodalInfo
from ..data_item import CacheItem
from ..mllm_tokenize_fn.qwen3_vl_tokenize_fn import Qwen3VLTokenizeFnConfig, Qwen3VLTokenizeFunction, QwenVL3DataItem
from ..utils import replace_image_context_and_collect_media_data


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


class RLQwen3VLTokenizeFunction(Qwen3VLTokenizeFunction):
    def __init__(self, *args, ignore_multimodal_info: bool = False, **kwargs):
        self.ignore_multimodal_info = ignore_multimodal_info
        super().__init__(*args, **kwargs)

    # TODO: tool call
    def __call__(self, item: dict, media_root: str = "", **kwargs) -> RolloutState | CacheItem:
        extra_info = item.get("extra_info", {})
        message = item["prompt"]

        data = super().__call__({"messages": message}, media_root=media_root)

        if self.state == "cache":
            return {
                "num_tokens": data["num_tokens"],
                "proxy_attn_flops": data.get("proxy_attn_flops", float(data["num_tokens"])),
            }
        else:
            data = cast(QwenVL3DataItem, data)
            image_data, _ = replace_image_context_and_collect_media_data(message, media_root, True)
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
            return RolloutState(
                message=message,
                num_tokens=data["num_tokens"],
                proxy_attn_flops=data.get("proxy_attn_flops", float(data["num_tokens"])),
                prompt_ids=prompt_token_ids,
                position_ids=data["position_ids"],
                data_source=item.get("data_source", "default"),
                reward_model=item.get("reward_model", {}),
                mm_info=mm_info,
                extra_fields=extra_info,
            )

    def hash(self) -> str:
        return "RLQwen3VLTokenizeFunction"


class RLQwen3VLTokenizeFnConfig(Qwen3VLTokenizeFnConfig):
    ignore_multimodal_info: bool = False  # eval is True

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
        )
