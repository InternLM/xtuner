from pathlib import Path
from typing import List, Literal, Optional, Union

from cyclopts import Group, Parameter
from pydantic import BaseModel
from typing_extensions import Annotated


worker_group = Group("worker", help="Types of workers available.")
train_group = Group("Training", sort_key=90, help="Training worker configuration.")
infer_group = Group("inference", help="Inference worker configuration.")


class TrainingWorkerConfig(BaseModel):
    """Configuration for the TraingWorker."""

    type: Literal["train"] = "train"
    train_model_path: Annotated[str, Parameter(group=train_group, help="Path to the training model.")]


class RolloutConfig(BaseModel):
    # base config
    env: Annotated[
        str,
        Parameter(group=infer_group, help="Environment variables to set for the rollout."),
    ] = ""
    backend: Annotated[
        str,
        Parameter(group=infer_group, help="Backend framework for the rollout worker, e.g., 'vllm', 'lmdeploy'."),
    ] = "lmdeploy"
    model_path: Annotated[str | Path, Parameter(group=infer_group, help="Path to the SGLang model.")] = ""
    model_name: Annotated[str, Parameter(group=infer_group, help="Name of the model to be used in the LMDeploy.")] = ""
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


if __name__ == "__main__":
    from cyclopts import App, Group, Parameter

    app = App()

    @app.default
    def test_command(*, config: RolloutConfig):
        """A test command to verify the command line interface."""
        print("This is a test command.")

    app()
