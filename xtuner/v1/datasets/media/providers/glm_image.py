from .shared import CertifiedImageProvider


class GlmImageProvider(CertifiedImageProvider):
    name = "glm_image"
    processor_module_prefix = "transformers.models.glm5_next."
    native_per_image = True

    def _preprocess(self, images):
        return self.processor(images=images, return_tensors="pt")
