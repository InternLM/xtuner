import os
from typing import Dict, List, Literal, Optional, Union

from cyclopts import Group, Parameter, validators
from pydantic import BaseModel
from typing_extensions import Annotated


worker_group = Group("worker", help="Types of workers available.")


if os.getenv("XTUNER_USE_SGLANG", "0") == "1":
    sglang_group = Group("SGLang", sort_key=100, show=True, help="SGLang worker configuration.")
else:
    sglang_group = Group("SGLang", sort_key=100, show=False, help="SGLang worker configuration.")


sglang_group = Group("SGLang", sort_key=110, help="SGLang worker configuration.")


class SGLangWorkerConfig(BaseModel):
    """Configuration for the SGLangWorker."""

    model_path: Annotated[str, Parameter(group=sglang_group, help="Path to the SGLang model.")]


if os.getenv("XTUNER_USE_SGLANG", "0") == "1":
    vllm_group = Group("vLLM", sort_key=100, show=True, help="vLLM worker configuration.")
else:
    vllm_group = Group("vLLM", sort_key=100, show=False, help="vLLM worker configuration.")


class VLLMWorkerConfig(BaseModel):
    """Configuration for the vLLMWorker."""


train_group = Group("Training", sort_key=90, help="Training worker configuration.")


class TrainingWorkerConfig(BaseModel):
    """Configuration for the TraingWorker."""

    type: Literal["train"] = "train"
    train_model_path: Annotated[str, Parameter(group=train_group, help="Path to the training model.")]


if os.getenv("XTUNER_USE_LMDEPLOY", "0") == "1":
    from lmdeploy import ChatTemplateConfig, PytorchEngineConfig, TurbomindEngineConfig

    lmdeploy_group = Group("LMDeploy", sort_key=100, show=True, help="LMDeploy Worker Configuration.")
    lmdeploy_pytorch_group = Group(
        "LMDeploy Pytorch Engine", sort_key=100, show=True, help="LMDeploy PyTorch Engine Configuration."
    )
    lmdeploy_turbomind_group = Group(
        "LMDeploy Turbomind Engine", sort_key=100, show=True, help="LMDeploy TurboMind Engine Configuration."
    )
else:
    ChatTemplateConfig = None
    PytorchEngineConfig = None
    TurbomindEngineConfig = None
    lmdeploy_group = Group("LMDeploy", sort_key=100, show=False, help="LMDeploy Worker Configuration.")
    lmdeploy_pytorch_group = Group(
        "LMDeploy Pytorch Engine", sort_key=100, show=False, help="LMDeploy PyTorch Engine Configuration."
    )
    lmdeploy_turbomind_group = Group(
        "LMDeploy Turbomind Engine", sort_key=100, show=False, help="LMDeploy TurboMind Engine Configuration."
    )


# lmdeploy_pytorch_group = Group("lmdeploy pytorch", sort_key=100, help="LMDeploy PyTorch Engine configuration.")
# lmdeploy_turbomind_group = Group("lmdeploy turbomind", sort_key=100, help="LMDeploy TurboMind Engine configuration.")
# lmdeploy_backend_group = Group(
#     "lmdeploy backend", sort_key=100, validator=validators.MutuallyExclusive, help="LMDeploy backend configuration."
# )


class LMDeployBackendConfig(BaseModel):
    pytorch: Annotated[
        Optional[PytorchEngineConfig],
        Parameter(group=lmdeploy_pytorch_group, help="Configuration for the PyTorch backend of the LMDeploy."),
    ] = None
    turbomind: Annotated[
        Optional[TurbomindEngineConfig],
        Parameter(group=lmdeploy_turbomind_group, help="Configuration for the TurboMind backend of the LMDeploy."),
    ] = None


class LMDeployWorkerConfig(BaseModel):
    """Configuration for the LMDeploy worker."""

    model_path: Annotated[
        str, Parameter(group=lmdeploy_group, help="Path to the model to be used in the LMDeploy.")
    ] = ""
    model_name: Annotated[
        str, Parameter(group=lmdeploy_group, help="Name of the model to be used in the LMDeploy.")
    ] = ""
    backend: Annotated[
        LMDeployBackendConfig,
        Parameter(help="Backend to use for the LMDeploy engine, e.g., 'pytorch'"),
    ]

    # backend_config: Annotated[PytorchEngineConfig | TurbomindEngineConfig | None, Parameter(
    #     help="Configuration for the LMDeploy backend engine."
    # )] = None
    chat_template: Annotated[
        Optional[ChatTemplateConfig],
        Parameter(group=lmdeploy_group, help="Configuration for the chat template used in the LMDeploy."),
    ] = None
    log_level: Annotated[str, Parameter(group=lmdeploy_group, help="Logging level for the LMDeploy service.")] = "WARN"
    api_key: Annotated[
        Optional[Union[List[str], str]],
        Parameter(
            group=lmdeploy_group, help="API keys for the LMDeploy service. Can be a single key or a list of keys."
        ),
    ] = None
    max_log_len: Annotated[
        Optional[int],
        Parameter(group=lmdeploy_group, help="Max number of prompt characters or prompt tokens being printed in log."),
    ] = None
    reasoning_parser: Annotated[Optional[str], Parameter(group=lmdeploy_group, help="The reasoning parser name.")] = (
        None
    )
    tool_call_parser: Annotated[Optional[str], Parameter(group=lmdeploy_group, help="The tool call parser name.")] = (
        None
    )
    server_name: Annotated[str, Parameter(group=lmdeploy_group, help="ip address of the LMDeploy server.")] = "0.0.0.0"
    server_port: Annotated[int, Parameter(group=lmdeploy_group, help="Port number of the LMDeploy server.")] = 23333
    env: Annotated[
        Optional[Dict[str, str]],
        Parameter(group=lmdeploy_group, help="Environment variables to set for the LMDeploy."),
    ] = None


infer_group = Group("inference", validator=validators.MutuallyExclusive(), help="Inference worker configuration.")


class InfrerenceWorkerConfig(BaseModel):
    """Configuration for the InferenceWorker."""

    lmdeploy: Annotated[
        Optional[LMDeployWorkerConfig], Parameter(group=infer_group, help="Configuration for the LMDeploy worker.")
    ] = None

    vllm: Annotated[
        Optional[VLLMWorkerConfig], Parameter(group=infer_group, help="Configuration for the VLLM worker.")
    ] = None

    sglang: Annotated[
        Optional[SGLangWorkerConfig], Parameter(group=infer_group, help="Configuration for the SGLang worker.")
    ] = None
