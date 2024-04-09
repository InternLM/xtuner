# Copyright (c) OpenMMLab. All rights reserved.
import logging
from abc import abstractclassmethod
from typing import Dict, List, Literal, Optional, Sequence, Union

from mmengine import print_log
from mmengine.model import BaseModel

from xtuner.chat.streamer import SteamerType
from xtuner.types import BaseMessages, ChatBackendProtocol, SampleParams
from .utils import guess_load_checkpoint


class BaseAlgorithm(BaseModel, ChatBackendProtocol):
    """The base class for all training algorithms in XTuner.

    All algorithms developed based on XTuner should inherit from this class
    and implement the abstract methods in this class.

    As long as the subclass inherits and rewrites the abstract methods in this
    class, it can reuse various tool chains provided in XTuner, such as
    automatic loading of models, model inference deployment etc.

    This class inherits from `mmengine.model.BaseModel` for using the
    training capabilities provided by MMEngine; it also inherits from
    `xtuner.types.ChatBackendProtocol` for access to the tool chains related
    to deployment and evaluation.

    Note:
        After building the model, if `_is_init` is False, MMEngine will call
        `init_weights()` to initialize the model parameters. Since the LLM
        always loads pre-trained weights during training, to avoid the weights
        being overwritten, the model initialization function is turned off by
        default in the base class. If there is an algorithm that requires
        model initialization, you can call the `enable_init_weights()`
        provided by the base class and implement the corresponding
        initialization logic in `init_weights()`
    """

    def __init__(self) -> None:
        """Initialize the BaseAlgorithm."""
        super().__init__()
        self._is_init = True

    def init_weights(self) -> None:
        """An interface required by MMEngine.

        This method allows customizing the logic of model initialization. If
        not defined, the default initialization method of torch will be used.

        This will be called by MMEngine Runner before training starts. If
        you're not using MMEngine Runner for training, you need to explicitly
        call this method.
        """
        pass

    def enable_init_weights(self) -> None:
        """Set the `_is_init` flag to False to enable initializing the
        weight."""
        self._is_init = False

    def gradient_checkpointing_enable(self) -> None:
        """Define how to enable gradient checkpointing.

        Note:
            When overloading this method, correspondingly overload
            `gradient_checkpointing_disable`.
        """

        msg = (f'{type(self)} has not implemented '
               '`gradient_checkpoint_enable()`, which may consume a lot of '
               'GPU memory. If you want to reduce GPU memory usage, please '
               f'override `gradient_checkpoint_enable()` in {type(self)}.')
        print_log(msg, logger='current', level=logging.WARNING)

    def gradient_checkpointing_disable(self) -> None:
        """Define how to disable gradient checkpointing.

        Note:
            When overloading this method, correspondingly overload
            `gradient_checkpointing_enable`.
        """
        msg = (f'{type(self)} has not implemented '
               '`gradient_checkpoint_disable()`.')
        print_log(msg, logger='current', level=logging.WARNING)

    def chat(self,
             prompt_or_messages: Union[str, BaseMessages],
             sample_params: Optional[SampleParams] = None,
             streamer: Optional[SteamerType] = None) -> str:
        """Define the action when receiving a chat message.

        This interface should be consistent with `ChatBackendProtocol` for
        easier integration with other modules of XTuner."

        Args:
            prompt_or_messages (str | BaseMessages):
                Prompt or messages in OpenAI Format
            sample_params (SampleParams | None):
                The hyperparameters controlling the generation results of the
                model, if set to None, the model will generate according to
                its default behavior. Default is None.
            streamer (SteamerType | None):
                The mode of streaming output can be controlled by different
                streamers. If set to None, it is non-streaming output.
                Default is None.

        Returns:
            The response of the model to the input messages should be a string.
        """

        raise NotImplementedError(f'{type(self)} has not implemented the '
                                  '`chat` interface. Please refer to '
                                  'the interface conventions in '
                                  '`ChatBackendProtocol` and implement the '
                                  f'`chat` interface in {type(self)}')

    def batch_infer(self,
                    prompt_or_messages_list: Union[str, BaseMessages],
                    sample_params: Optional[SampleParams] = None,
                    streamer: Optional[SteamerType] = None) -> List[str]:
        """Define the batch inference routine.

        This interface should be consistent with `ChatBackendProtocol` for
        easier integration with other modules of XTuner."

        Note:
            `SampleParams` is a data structure defined by XTuner.
            When inheriting and overriding this method, developers need to
            convert `sample_params` into the required data structure on their
            own. For instance, when using a Huggingface model, it should be
            converted to `GenerationConfig`.

        Args:
            prompt_or_messages_list (Union[str, BaseMessages]):
                Multiple messages in OpenAI format or multiple prompts;
            sample_params (SampleParams | None):
                The hyperparameters controlling the generation results of the
                model, if set to None, the model will generate according to
                its default behavior. Default is None.
            streamer (SteamerType | None):
                The mode of streaming output can be controlled by different
                streamers. If set to None, it is non-streaming output.
                Default is None.

        Returns:
            The model responds to multiple messages, the result should be a
            list of strings.
        """
        raise NotImplementedError(f'{type(self)} has not implemented the '
                                  '`batch_infer` interface. Please refer to '
                                  'the interface conventions in '
                                  '`ChatBackendProtocol` and implement the '
                                  f'`batch_infer` interface in {type(self)}')

    def save_pretrained(self, save_dir: str, config: str):
        """Define how to save a model that can be loaded with `from_pretrained`

        Note:
            Unlike checkpoints, `save pretrained` does not require the saving
            of information such as the optimizer's state.

        Note:
            The `save_pretrained` and `from_pretrained` depend on each other,
            there are no strict requirements, they just need to be mutually
            compatible.

        Note:
            `save_pretrained` and `from_pretrained` are meant to unify the
            interface, making it convenient to automatically load the model
            through `AutoXTuner.from_pretrained`. If these methods are not
            implemented, it won't affect the training-related features. If
            this feature isn't needed, there is no need to override this
            method.

        Args:
            save_dir (str): Directory where the model is saved.
            config (str): The path of the config file used during training.
        """

        raise NotImplementedError(f'{type(self)} has not implemented the '
                                  '`save_pretrained` interface. Please refer '
                                  'to the interface conventions in '
                                  '`BaseAlgorithm` and implement the '
                                  '`save_pretrained` interface in '
                                  f'{type(self)}')

    @classmethod
    def from_pretrained(
        self,
        model_path_or_id: str,
        config: Optional[str] = None,
        from_hub: Literal['huggingface',
                          'modelscope'] = 'huggingface') -> None:
        """Define how to load a model saved with `save_pretrained`.

        Note:
            The `save_pretrained` and `from_pretrained` depend on each other,
            there are no strict requirements, they just need to be mutually
            compatible.

        Note:
            `save_pretrained` and `from_pretrained` are meant to unify the
            interface, making it convenient to automatically load the model
            through `AutoXTuner.from_pretrained`. If these methods are not
            implemented, it won't affect the training-related features. If
            this feature isn't needed, there is no need to override this
            method.

        Args:
            model_path_or_id (str): The model id or model path.
            config (str | None): The config path. Default is None.
            from_hub (str): The model hosting hub, modelscope, or huggingface.
                Default is huggingface.

        Raises:
            RuntimeError:
                When model_path_or_id does not contain the xtuner's config
                file and the input config is None, a RuntimeError should be
                thrown.
        """

        raise NotImplementedError(f'{type(self)} has not implemented the '
                                  '`from_pretrained` interface. Please refer '
                                  'to the interface conventions in '
                                  '`BaseAlgorithm` and implement the '
                                  '`from_pretrained` interface in '
                                  f'{type(self)}')

    def load_checkpoint(self, checkpoint: str):
        """Load the checkpoint in the XTuner training process.

        Because different parallel strategies (Pytorch DDP or DeepSpeed
        ZeRO1/2/3) may be used during training, the checkpoint formats saved
        by different training strategies are not exactly the same.

        `guess_load_checkpoint` will automatically recognize the checkpoint
        format and load it.

        Note:
            It is important to note that during the training process, the
            checkpoint saves the optimizer state, which can result in file
            sizes multiple times larger than the model weights. To facilitate
            delivery or share the trained model with the community, you should
            save a checkpoint that only includes the model weights using
            `save_checkpoint`.
        """
        state_dict = guess_load_checkpoint(checkpoint)
        self.load_state_dict(state_dict)

    @abstractclassmethod
    def dataloader_collate_fn(cls, instances: Sequence) -> Dict:
        """Define how to collate the data fetched from the dataloader.

        To adapt to the training process of MMEngine, this method should
        return a dictionary that contains two fields, `data` and
        `data_samples`, corresponding to `data` and `data_samples` in the
        `forward` method.

        In accordance with the conventions of MMEngine, `data` should contain
        training-relevant content, such as `input_ids` and `labels`;
        `data_samples` can be some record-type data, convenient for logging or
        visual analysis, such as the prompt of the original data, etc.

        Note:
            This is one of the arguments of `torch.utils.data.Dataloader`,
            mainly used for data preprocessing.

            To clearly illustrate the data format of each algorithm's input,
            XTuner makes `collate_fn` as an abstract method that each
            algorithm must implement.

            It should be noted that this method is a `classmethod`, which can
            be called without instantiating an object.

        Args:
            instances: The original data fetched from the dataloader. The
            number of instances is consistent with the batch_size of the
            dataloader.

        Returns:
            It must be a dictionary containing the `data` and `data_samples`
            fields. `data_samples` can be None, but this field cannot be
            absent.
        """


class BaseTune(BaseAlgorithm):
    pass
