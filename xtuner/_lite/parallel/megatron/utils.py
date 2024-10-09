def map_rank0_modules(model, rank0_model):
    rank0_modules = {name: mod for name, mod in rank0_model.named_modules()}
    rank0_map = {
        mod: rank0_modules[name]
        for name, mod in model.named_modules()
    }
    return rank0_map
