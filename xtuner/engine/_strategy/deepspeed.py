from mmengine._strategy import DeepSpeedStrategy as MMEngineDeepSpeedStrategy

from xtuner.registry import STRATEGIES


@STRATEGIES.register_module()
class DeepSpeedStrategy(MMEngineDeepSpeedStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from transformers.integrations.deepspeed import HfDeepSpeedConfig

        # hf_deepspeed_config has to be saved as an attribute.
        self.hf_deepspeed_config = HfDeepSpeedConfig(self.config)
