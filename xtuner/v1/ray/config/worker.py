import json
import os
import socket
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
    device: Annotated[str, Parameter(group=infer_group, help="Device to be used for the rollout worker.")] = "GPU"
    model_path: Annotated[str | Path, Parameter(group=infer_group, help="Path to the SGLang model.")]
    model_name: Annotated[
        str | None, Parameter(group=infer_group, help="Name of the model to be used in the LMDeploy.")
    ] = None
    tokenizer_path: Annotated[
        str | None, Parameter(group=infer_group, help="Path to the tokenizer for the model.")
    ] = None
    api_key: Annotated[
        Optional[Union[List[str], str]],
        Parameter(
            group=infer_group,
            help="API keys for the rollout service. Can be a single key or a list of keys.",
        ),
    ] = None
    api_port: Annotated[
        Optional[int],
        Parameter(group=infer_group, help="Port number for the rollout API server. If not set, 8000 will be used."),
    ] = 8000
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
    rollout_max_batch_size: Annotated[
        Optional[int],
        Parameter(
            group=infer_group,
            help="Maximum batch size for the rollout worker. If not set, it will be determined automatically based on the model and GPU memory.",
        ),
    ] = None
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
    context_length: Annotated[
        Optional[int],
        Parameter(
            group=infer_group,
            help="Context length for the rollout worker.",
        ),
    ] = None
    extra_rollout_config: Annotated[
        dict,
        Parameter(
            group=infer_group,
            help='Extra configuration for different rollout worker. vllm parameters will start with prefix "vllm", etc.',
        ),
    ] = {"lmdeploy_log_level": "CRITICAL", "lmdeploy_uvicorn_log_level": "CRITICAL"}
    worker_log_dir: Annotated[Path, Parameter(help="Directory to save worker logs.")] = Path.cwd() / "work_dir"

    def __init__(self, **kwargs):
        if "model_name" not in kwargs:
            model_name_from_config = None
            model_path = Path(kwargs["model_path"])
            config_json_path = model_path / "config.json"
            try:
                with open(config_json_path, encoding="utf-8") as f:
                    config_data = json.load(f)
                    model_name_from_config = config_data.get("model_type")
            except (json.JSONDecodeError, OSError):
                pass

            if model_name_from_config:
                kwargs["model_name"] = model_name_from_config
            else:
                kwargs["model_name"] = model_path.name

        if "tokenizer_path" not in kwargs:
            kwargs["tokenizer_path"] = str(kwargs["model_path"])

        port = kwargs.get("api_port", 8000)
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("localhost", port))
                    break
                except OSError:
                    port += 1
        kwargs["api_port"] = port

        if "device" in kwargs and kwargs["device"] == "NPU":
            kwargs["gpus_per_node"] = 16

        rollout_backend = ""
        if os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
            rollout_backend = "sglang"
        elif os.environ.get("XTUNER_USE_VLLM", "0") == "1":
            rollout_backend = "vllm"
        elif os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "1":
            rollout_backend = "lmdeploy"

        assert rollout_backend in ["sglang", "vllm", "lmdeploy"], (
            f"Unsupported rollout backend: {rollout_backend}. Please set XTUNER_USE_SGLANG, XTUNER_USE_VLLM, or XTUNER_USE_LMDEPLOY to 1."
        )
        if rollout_backend == "sglang":
            kwargs["launch_server_method"] = "multiprocessing"
            kwargs["rollout_cross_node_comm"] = False
        else:
            kwargs["launch_server_method"] = "ray"
            kwargs["rollout_cross_node_comm"] = True

        super().__init__(**kwargs)
        self.worker_log_dir.mkdir(parents=True, exist_ok=True)


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
