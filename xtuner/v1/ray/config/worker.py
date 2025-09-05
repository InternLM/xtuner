from pathlib import Path
from typing import List, Literal, Optional, Union

from cyclopts import Group, Parameter
from pydantic import BaseModel
from typing_extensions import Annotated


worker_group = Group("worker", help="Types of workers available.")
train_group = Group("Training", sort_key=90, help="Training worker configuration.")
infer_group = Group("inference", help="Inference worker configuration.")


class TrainingWorkerConfig(BaseModel):
    """Configuration for the TrainingWorker."""

    type: Literal["train"] = "train"
    train_model_path: Annotated[str, Parameter(group=train_group, help="Path to the training model.")]


class RolloutConfig(BaseModel):
    """Rollout worker configuration for XTuner.

    This class defines comprehensive configuration parameters for rollout workers in XTuner,
    supporting multiple inference backends with distributed computing and optimization features.

    Args:
        env (str): Environment variables for the rollout worker. Defaults to "".
        backend (str): Backend framework ('vllm', 'lmdeploy', etc.). Defaults to "lmdeploy".
        model_path (str | Path): Path to the inference model.
        model_name (str): Model name for the backend engine.
        tokenizer_path (str): Path to the model tokenizer. Defaults to "".
        api_key (Optional[Union[List[str], str]]): API keys for rollout service.
            Supports single key or list of keys. Defaults to None.

        gpus_per_node (int): Number of GPUs per node. Defaults to 8.
        dtype (str): Model data type ('bfloat16', 'float16', 'int8'). Defaults to "bfloat16".
        gpu_memory_utilization (float): GPU memory utilization ratio. Defaults to 0.85.
        random_seed (int): Random seed for reproducible generation. Defaults to 1024.

        rollout_cross_node_comm (bool): Enable cross-node communication. Defaults to False.
        tensor_parallel_size (int): GPUs per inference engine (tensor parallelism). Defaults to 1.
        expert_parallel_size (int): Experts per inference engine (expert parallelism). Defaults to 1.

        enable_chunked_prefill (bool): Enable chunked prefill for memory efficiency. Defaults to False.
        chunked_prefill_size (int): Chunk size for prefill operations. Defaults to 128.
        skip_load_weights (bool): Skip weight loading for rollout worker. Defaults to False.
        rollout_timeout (float): Timeout duration in seconds for rollout requests. Defaults to 3600.0.

        launch_server_method (Literal["ray", "multiprocessing"]): Server launch method. Defaults to "ray".
        system_prompt (Optional[str]): System prompt to guide generation behavior. Defaults to None.
        extra_rollout_config (Optional[dict]): Backend-specific configurations using engine prefixes
            (e.g., 'vllm_enable_chunked_prefill', 'lmdeploy_max_batch_size'). Defaults to empty dict.

    **Examples:**

    Example configuration with LMDeploy backend::

        config = RolloutConfig(
            env="test_env",
            model_path="Qwen/Qwen3-8B",
            model_name="Qwen3-8B",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.6,
            gpus_per_node=8,
            backend="lmdeploy",
        )
    """

    # base config
    env: Annotated[
        str,
        Parameter(group=infer_group, help="Environment variables to set for the rollout."),
    ] = ""
    backend: Annotated[
        str,
        Parameter(group=infer_group, help="Backend framework for the rollout worker, e.g., 'vllm', 'lmdeploy'."),
    ] = "lmdeploy"
    model_path: Annotated[str | Path, Parameter(group=infer_group, help="Path to the SGLang model.")]
    model_name: Annotated[str, Parameter(group=infer_group, help="Name of the model to be used in the LMDeploy.")]
    tokenizer_path: Annotated[str, Parameter(group=infer_group, help="Path to the tokenizer for the model.")] = ""
    api_key: Annotated[
        Optional[Union[List[str], str]],
        Parameter(
            group=infer_group,
            help="API keys for the rollout service. Can be a single key or a list of keys.",
        ),
    ] = None
    gpus_per_node: Annotated[int, Parameter(group=infer_group, help="Number of GPUs allocated per node.")] = 8
    dtype: Annotated[
        str,
        Parameter(group=infer_group, help="Data type for the model, e.g., 'bfloat16', 'float16', 'int8'."),
    ] = "bfloat16"
    gpu_memory_utilization: Annotated[
        float, Parameter(group=infer_group, help="GPU memory utilization for the rollout worker.")
    ] = 0.85
    random_seed: Annotated[int, Parameter(group=infer_group, help="Random seed for the rollout worker.")] = 1024
    # distributed config
    rollout_cross_node_comm: Annotated[
        bool,
        Parameter(
            group=infer_group,
            help="Whether to enable cross-node communication for the rollout worker.",
        ),
    ] = False
    tensor_parallel_size: Annotated[
        int,
        Parameter(
            group=infer_group,
            help="Number of GPUs allocated for each inference engine in the rollout worker.",
        ),
    ] = 1
    expert_parallel_size: Annotated[
        int,
        Parameter(
            group=infer_group,
            help="Number of experts allocated for each inference engine in the rollout worker.",
        ),
    ] = 1
    # optimization config
    enable_chunked_prefill: Annotated[
        bool,
        Parameter(
            group=infer_group,
            help="Whether to enable chunked prefill for the rollout worker.",
        ),
    ] = False
    chunked_prefill_size: Annotated[
        int,
        Parameter(
            group=infer_group,
            help="Chunked prefill size for the rollout worker.",
        ),
    ] = 128
    skip_load_weights: Annotated[
        bool,
        Parameter(
            group=infer_group,
            help="Whether to skip loading weights for the rollout worker.",
        ),
    ] = False
    extra_rollout_config: Annotated[
        Optional[dict],
        Parameter(
            group=infer_group,
            help='Extra configuration for different rollout worker. vllm parameters will start with prefix "vllm", etc.',
        ),
    ] = dict()
    launch_server_method: Annotated[
        Literal["ray", "multiprocessing"],
        Parameter(
            group=infer_group,
            help="Method to launch the rollout server, either 'ray' or 'multiprocessing'.",
        ),
    ] = "ray"
    rollout_timeout: Annotated[
        float,
        Parameter(
            group=infer_group,
            help="Timeout duration (in seconds) for rollout requests.",
        ),
    ] = 3600.0
    system_prompt: Annotated[
        Optional[str],
        Parameter(
            group=infer_group,
            help="System prompt for the rollout worker.",
        ),
    ] = None
    enable_fp8: Annotated[
        bool,
        Parameter(
            group=infer_group,
            help="Whether to enable FP8 quantization for the rollout worker.",
        ),
    ] = False


if __name__ == "__main__":
    from cyclopts import App, Group, Parameter

    app = App()

    @app.default
    def test_command(*, config: RolloutConfig):
        """A test command to verify the command line interface.

        Args:
            config: The rollout configuration.
        """
        print("This is a test command.")

    app()
