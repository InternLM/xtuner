from .shared import CertifiedImageProvider


class QwenImageProvider(CertifiedImageProvider):
    name = "qwen_image"
    processor_module_prefix = "transformers.models.qwen"

    def _preprocess(self, images):
        return self.processor.preprocess(images, return_tensors="pt")
