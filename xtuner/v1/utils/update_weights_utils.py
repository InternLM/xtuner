from torch.multiprocessing import reductions


def monkey_unpatch_torch_reductions():
    # SGLang patches `torch.multiprocessing.reductions` for tensor serialization
    # (see: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/patch_torch.py#L25).
    # However, this patch only considers GPU tensors and causes issues when serializing
    # CPU tensors, which occurs when XTuner saves a model to the Hugging Face format.
    # This function reverts the monkey-patch to restore the original torch functionality,
    # ensuring that CPU tensors can be serialized correctly.
    if hasattr(reductions, "_reduce_tensor_original"):
        reductions.reduce_tensor = reductions._reduce_tensor_original
        reductions.rebuild_cuda_tensor = reductions._rebuild_cuda_tensor_original
        reductions.init_reductions()
