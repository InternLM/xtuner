from loguru import logger


def get_gpu_requirement(trainer_config: dict) -> int:
    # Calculates the number of GPUs required for a given trainer configuration.
    num_gpus = 1
    if 'parallel' in trainer_config:
        parallel = trainer_config['parallel']
        data = parallel.get('data', {'size': 1})
        tensor = parallel.get('tensor', {'size': 1})
        pipeline = parallel.get('pipeline', {'size': 1})
        num_gpus = data['size'] * tensor['size'] * pipeline['size']
    return num_gpus


def get_resource_requirement(model_configs: dict) -> dict:
    """Analyzes resource requirements for a list of model configs and returns a
    dictionary with the total number of GPUs and CPUs required.

    Args:
        model_configs (dict): A dictionary containing model configurations.

    Returns:
        dict: A dictionary with the total number of GPUs and CPUs required.
    """

    resources = {'num_gpus': 0}
    for name, model_config in model_configs.items():
        if 'trainer_config' not in model_config:
            logger.warning(f'{name} has no trainer_config. SKIP.')
            continue
        trainer_config = model_config['trainer_config']
        num_gpus = get_gpu_requirement(trainer_config)

        if 'generator_config' in model_config:
            generator_config = model_config['generator_config']
            if not generator_config.get(
                    'shared_with_trainer'):  # None or False
                num_gpus += get_gpu_requirement(generator_config)

        resources['num_gpus'] += num_gpus

    resources['num_cpus'] = resources['num_gpus'] * 10
    return resources


def get_dp_size(trainer_config: dict) -> int:
    dp_size = 1
    if 'parallel' in trainer_config:
        parallel = trainer_config['parallel']
        data = parallel.get('data', {'size': 1})
        dp_size = data['size']
    return dp_size


def get_tp_size(trainer_config: dict) -> int:
    tp_size = 1
    if 'parallel' in trainer_config:
        parallel = trainer_config['parallel']
        data = parallel.get('tensor', {'size': 1})
        tp_size = data['size']
    return tp_size


def get_pp_size(trainer_config: dict) -> int:
    pp_size = 1
    if 'parallel' in trainer_config:
        parallel = trainer_config['parallel']
        data = parallel.get('pipeline', {'size': 1})
        pp_size = data['size']
    return pp_size
