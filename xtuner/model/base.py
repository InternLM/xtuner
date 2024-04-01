# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractclassmethod, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

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

    To unify training and inference, this class includes the following seven
    abstract interfaces:

    1. `dataloader_collate_fn`
    This is one of the arguments of `torch.utils.data.Dataloader`, mainly used
    for data preprocessing.
    To clearly illustrate the data format of each algorithm's input, XTuner
    makes `collate_fn` a mandatory method that each algorithm must implement.
    It should be noted that this method is a `classmethod`, which can be
    called without instantiating an object.

    2. `gradient_checkpointing_enable`:
    This is a memory-reducing technique that clears activations of specific
    layers and recomputes them during a backward pass. When developing a new
    algorithm, developers should implement the logic of enabling gradient
    checkpointing according to the used model.
    If you do not want to use this feature, you can also define an empty
    function, but this way can consume a large amount of memory when training
    LLM models.

    3. `save_checkpoint`
    This defines how to save the weights of the trained model, supporting both
    HuggingFace format (to_hub=True) and ordinary PTH format (to_hub=False).
    If the algorithm only need one format, then the other format should throw
    a NotImplementedError.
    Please note this method does not change the format of checkpoints saved
    during the training process as these checkpoints are more complex
    including optimizer states.
    If you want to save to the checkpoint in this format, you need to load the
    model and specifically call this method.

    4. `load_checkpoint`
    This defines how to load the checkpoints saved by the `save_checkpoint`
    method, supporting both HuggingFace format (from_hub=True) and normal
    PTH format (from_hub=False).
    This method will not be called during the training process, but it will be
    called when `AutoXTuner.from_pretrained` is used. If this method is not
    implemented, the model cannot be automatically loaded.

    5. `chat`
    When developing a new algorithm, you must define how to converse with a
    trained model, whether for accuracy testing or model deployment. To ensure
    generality, the interface arguments should follow the
    `xtuern.types.ChatBackendProtocol`.
    If there's no need for XTuner's toolchain, you can directly throw a
    `NotImplementedError`.

    6. `get_logits`
    Obtain the logits corresponding to a `messages`; if there's no need for
    related functions that require logits (visualization, API server, etc.),
    you can directly throw a `NotImplementedError`.

    7. `batch_infer`
    Define how a trained model handles batch data. If the related function is
    not needed, you can directly throw a `NotImplementedError`.
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

    @abstractmethod
    def gradient_checkpointing_enable(self) -> None:
        """Define how to enable gradient checkpointing."""

    @abstractmethod
    def chat(self,
             messages: BaseMessages,
             sample_params: Optional[SampleParams] = None,
             streamer: Optional[SteamerType] = None) -> str:
        """Define the action when receiving a chat message.

        This interface should be consistent with `ChatBackendProtocol` for
        easier integration with other modules of XTuner."

        Args:
            messages (BaseMessages):
                History of Dialogues in OpenAI Format
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

    @abstractmethod
    def batch_infer(self, messages: List[BaseMessages],
                    sample_params: SampleParams, streamer: Any) -> List[str]:
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
            messages (List[BaseMessages]):
                Multiple historical dialogues in OpenAI format.
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

    @abstractmethod
    def save_checkpoint(self, save_dir: str, to_hub: bool = True) -> None:
        """Define how to save a Checkpoint.

        Args:
            save_dir (str): Directory where the checkpoint is saved".
            to_hub (bool):
                If True, the checkpoint should be saved in the
                Huggingface format.
                If False, the checkpoint should be saved in the standard
                PyTorch .pth format. Default is True.
        """

    def load_training_checkpoint(self, ckpt_dir: str):
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
        state_dict = guess_load_checkpoint(ckpt_dir)
        self.load_state_dict(state_dict)

    @abstractmethod
    def load_checkpoint(self,
                        ckpt_dir: str,
                        from_hub: bool = False) -> 'BaseAlgorithm':
        """Define how to load a checkpoint saved by `save_checkpoint`.

        Args:
            ckpt_dir (str):
                The directory where the checkpoint file is located.
            from_hub (bool):
                If True, the checkpoint will be loaded in the storage format
                used when to_hub is True in `save_checkpoint`.
                If False, the checkpoint will be loaded in the storage format
                used when to_hub is False in `save_checkpoint`.

        Note:
            When `from_hub` is set to False, it's necessary to be compatible
            with the checkpoint generated during XTuner training. To
            facilitate development, a `load_training_checkpoint` interface is
            provided in the base class.

        Note:
            It is important to note that during the training process, the
            checkpoint saves the optimizer state, which can result in file
            sizes multiple times larger than the model weights. To facilitate
            delivery or share the trained model with the community, you should
            save a checkpoint that only includes the model weights using
            `save_checkpoint`.
        """

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
