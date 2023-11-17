from mmengine._strategy import DeepSpeedStrategy as MMEngineDeepSpeedStrategy


class DeepSpeedStrategy(MMEngineDeepSpeedStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from transformers.integrations.deepspeed import HfDeepSpeedConfig
        _ = HfDeepSpeedConfig(self.config)
