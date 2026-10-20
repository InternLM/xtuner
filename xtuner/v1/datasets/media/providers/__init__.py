from ..provider import register_provider
from .shared import resolve_mixed_media_items


register_provider("glm_image", "xtuner.v1.datasets.media.providers.glm_image:GlmImageProvider")
register_provider("qwen_image", "xtuner.v1.datasets.media.providers.qwen_image:QwenImageProvider")
