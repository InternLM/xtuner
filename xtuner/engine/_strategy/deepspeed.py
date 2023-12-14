from mmengine._strategy import DeepSpeedStrategy as MMEngineDeepSpeedStrategy

from xtuner import DEEPSPEED_PETREL_ON
from xtuner.utils.fileio import patch_fileio


class DeepSpeedStrategy(MMEngineDeepSpeedStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from transformers.integrations.deepspeed import HfDeepSpeedConfig

        # hf_deepspeed_config has to be saved as an attribute.
        self.hf_deepspeed_config = HfDeepSpeedConfig(self.config)

    def _wrap_model(self, model):
        wrapper = super()._wrap_model(model)
        # hard code for deepspeed zero3
        # When utilizing Zero3, the model isn't allocated to CUDA within the
        # `deepspeed.initialize` process.
        assert hasattr(wrapper.model, 'data_preprocessor')
        wrapper.model.data_preprocessor.cuda()
        return wrapper

    def save_checkpoint(self, *args, **kwargs) -> None:
        if DEEPSPEED_PETREL_ON:
            with patch_fileio():
                super().save_checkpoint(*args, **kwargs)
        else:
            super().save_checkpoint(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs) -> None:
        if DEEPSPEED_PETREL_ON:
            with patch_fileio():
                super().load_checkpoint(*args, **kwargs)
        else:
            super().load_checkpoint(*args, **kwargs)

    def resume(self, *args, **kwargs) -> None:
        if DEEPSPEED_PETREL_ON:
            with patch_fileio():
                super().resume(*args, **kwargs)
        else:
            super().resume(*args, **kwargs)
