from torch.multiprocessing import reductions


def monkey_unpatch_torch_reductions():
    reductions.reduce_tensor = reductions._reduce_tensor_original
    reductions.rebuild_cuda_tensor = reductions._rebuild_cuda_tensor_original

    reductions.init_reductions()
