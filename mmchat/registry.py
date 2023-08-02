# Copyright (c) OpenMMLab. All rights reserved.
"""mmchat provides 21 registry nodes to support using modules across projects.
Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry

__all__ = [
    'RUNNERS', 'RUNNER_CONSTRUCTORS', 'LOOPS', 'HOOKS', 'LOG_PROCESSORS',
    'OPTIMIZERS', 'OPTIM_WRAPPERS', 'OPTIM_WRAPPER_CONSTRUCTORS',
    'PARAM_SCHEDULERS', 'DATASETS', 'DATA_SAMPLERS', 'TRANSFORMS', 'MODELS',
    'MODEL_WRAPPERS', 'WEIGHT_INITIALIZERS', 'BATCH_AUGMENTS', 'TASK_UTILS',
    'METRICS', 'EVALUATORS'
]

#######################################################################
#                         mmchat.engine                           #
#######################################################################

# Runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    'runner',
    parent=MMENGINE_RUNNERS,
    locations=['mmchat.engine'],
)
# Runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=MMENGINE_RUNNER_CONSTRUCTORS,
    locations=['mmchat.engine'],
)
# Loops which define the training or test process, like `EpochBasedTrainLoop`
LOOPS = Registry(
    'loop',
    parent=MMENGINE_LOOPS,
    locations=['mmchat.engine'],
)
# Hooks to add additional functions during running, like `CheckpointHook`
HOOKS = Registry(
    'hook',
    parent=MMENGINE_HOOKS,
    locations=['mmchat.engine'],
)
# Log processors to process the scalar log data.
LOG_PROCESSORS = Registry(
    'log processor',
    parent=MMENGINE_LOG_PROCESSORS,
    locations=['mmchat.engine'],
)
# Optimizers to optimize the model weights, like `SGD` and `Adam`.
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    locations=['mmchat.engine'],
)
# Optimizer wrappers to enhance the optimization process.
OPTIM_WRAPPERS = Registry(
    'optimizer_wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    locations=['mmchat.engine'],
)
# Optimizer constructors to customize the hyperparameters of optimizers.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=['mmchat.engine'],
)
# Parameter schedulers to dynamically adjust optimization parameters.
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    locations=['mmchat.engine'],
)

#######################################################################
#                        mmchat.datasets                          #
#######################################################################

# Datasets like `ImageNet` and `CIFAR10`.
DATASETS = Registry(
    'dataset',
    parent=MMENGINE_DATASETS,
    locations=['mmchat.datasets'],
)
# Samplers to sample the dataset.
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMENGINE_DATA_SAMPLERS,
    locations=['mmchat.datasets'],
)
# Transforms to process the samples from the dataset.
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['mmchat.datasets'],
)

#######################################################################
#                         mmchat.models                           #
#######################################################################

# Neural network modules inheriting `nn.Module`.
MODELS = Registry(
    'model',
    parent=MMENGINE_MODELS,
    locations=['mmchat.models'],
)
# Model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=['mmchat.models'],
)
# Weight initialization methods like uniform, xavier.
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=['mmchat.models'],
)
# Batch augmentations like `Mixup` and `CutMix`.
BATCH_AUGMENTS = Registry(
    'batch augment',
    locations=['mmchat.models'],
)
# Task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util',
    parent=MMENGINE_TASK_UTILS,
    locations=['mmchat.models'],
)
# Tokenizer to encode sequence
TOKENIZER = Registry(
    'tokenizer',
    locations=['mmchat.models'],
)

# Large Language Model
LLM = Registry(
    'llm',
    locations=['mmchat.models'],
)

#######################################################################
#                       mmchat.evaluation                         #
#######################################################################

# Metrics to evaluate the model prediction results.
METRICS = Registry(
    'metric',
    parent=MMENGINE_METRICS,
    locations=['mmchat.evaluation'],
)
# Evaluators to define the evaluation process.
EVALUATORS = Registry(
    'evaluator',
    parent=MMENGINE_EVALUATOR,
    locations=['mmchat.evaluation'],
)
