import io
from contextlib import contextmanager

import mmengine.fileio as fileio
from mmengine.fileio import LocalBackend, PetrelBackend, get_file_backend


def patch_func(module, fn_name_to_wrap):
    backup = getattr(patch_func, '_backup', [])
    fn_to_wrap = getattr(module, fn_name_to_wrap)

    def wrap(fn_new):
        setattr(module, fn_name_to_wrap, fn_new)
        backup.append((module, fn_name_to_wrap, fn_to_wrap))
        setattr(fn_new, '_fallback', fn_to_wrap)
        setattr(patch_func, '_backup', backup)
        return fn_new

    return wrap


@contextmanager
def patch_fileio(global_vars=None):
    if getattr(patch_fileio, '_patched', False):
        # Only patch once, avoid error caused by patch nestly.
        yield
        return
    import builtins

    @patch_func(builtins, 'open')
    def open(file, mode='r', *args, **kwargs):
        backend = get_file_backend(file)
        if isinstance(backend, LocalBackend):
            return open._fallback(file, mode, *args, **kwargs)
        if 'b' in mode:
            return io.BytesIO(backend.get(file, *args, **kwargs))
        else:
            return io.StringIO(backend.get_text(file, *args, **kwargs))

    if global_vars is not None and 'open' in global_vars:
        bak_open = global_vars['open']
        global_vars['open'] = builtins.open

    import os

    @patch_func(os.path, 'join')
    def join(a, *paths):
        backend = get_file_backend(
            a.decode('utf-8') if isinstance(a, bytes) else a)
        if isinstance(backend, LocalBackend):
            return join._fallback(a, *paths)
        paths = [item.lstrip('./') for item in paths if len(item) > 0]
        return backend.join_path(a, *paths)

    @patch_func(os.path, 'isdir')
    def isdir(path):
        backend = get_file_backend(path)
        if isinstance(backend, LocalBackend):
            return isdir._fallback(path)

        return backend.isdir(path)

    @patch_func(os.path, 'isfile')
    def isfile(path):
        backend = get_file_backend(path)
        if isinstance(backend, LocalBackend):
            return isfile._fallback(path)

        return backend.isfile(path)

    @patch_func(os.path, 'exists')
    def exists(path):
        backend = get_file_backend(path)
        if isinstance(backend, LocalBackend):
            return exists._fallback(path)
        return backend.exists(path)

    @patch_func(os, 'mkdir')
    def mkdir(path, *args, **kwargs):
        backend = get_file_backend(path)
        if isinstance(backend, LocalBackend):
            return mkdir._fallback(path, *args, **kwargs)

    @patch_func(os, 'makedirs')
    def makedirs(path, *args, **kwargs):
        backend = get_file_backend(path)
        if isinstance(backend, LocalBackend):
            return makedirs._fallback(path, *args, **kwargs)

    @patch_func(os, 'listdir')
    def listdir(path):
        backend = get_file_backend(path)
        if isinstance(backend, LocalBackend):
            return listdir._fallback(path)
        return backend.list_dir_or_file(path)

    @patch_func(os, 'chmod')
    def chmod(path, *args, **kwargs):
        backend = get_file_backend(path)
        if isinstance(backend, LocalBackend):
            return chmod._fallback(path, *args, **kwargs)

    @patch_func(os, 'stat')
    def stat(path, *args, **kwargs):
        backend = get_file_backend(path)
        if isinstance(backend, LocalBackend):
            return stat._fallback(path, *args, **kwargs)

    import glob as glob_pkg

    @patch_func(glob_pkg, 'glob')
    def glob(pathname, *, recursive=False):
        backend = get_file_backend(pathname)
        if isinstance(backend, LocalBackend):
            return glob._fallback(pathname, recursive=recursive)

        if pathname.endswith('*_optim_states.pt'):
            import os
            pathname = os.path.split(pathname)[0]
            files = backend.list_dir_or_file(pathname, recursive=recursive)
            files = [
                os.path.join(pathname, f) for f in files
                if f.endswith('_optim_states.pt')
            ]
        elif pathname.endswith('*_model_states.pt'):
            import os
            pathname = os.path.split(pathname)[0]
            files = backend.list_dir_or_file(pathname, recursive=recursive)
            files = [
                os.path.join(pathname, f) for f in files
                if f.endswith('_model_states.pt')
            ]
        elif '*' in pathname:
            raise NotImplementedError
        else:
            files = backend.list_dir_or_file(pathname, recursive=recursive)

        return files

    import filecmp

    @patch_func(filecmp, 'cmp')
    def cmp(f1, f2, *args, **kwargs):
        with fileio.get_local_path(f1) as f1, fileio.get_local_path(f2) as f2:
            return cmp._fallback(f1, f2, *args, **kwargs)

    import shutil

    @patch_func(shutil, 'copy')
    def copy(src, dst, **kwargs):
        from pathlib import Path

        if isinstance(src, Path):
            src = str(src).replace(':/', '://')
        if isinstance(dst, Path):
            dst = str(dst).replace(':/', '://')

        src_backend = get_file_backend(src)
        dst_backend = get_file_backend(dst)

        if isinstance(src_backend, LocalBackend) and isinstance(
                dst_backend, LocalBackend):
            return copy._fallback(src, dst, **kwargs)
        elif isinstance(src_backend, LocalBackend) and isinstance(
                dst_backend, PetrelBackend):
            return dst_backend.copyfile_from_local(str(src), str(dst))
        elif isinstance(src_backend, PetrelBackend) and isinstance(
                dst_backend, LocalBackend):
            return src_backend.copyfile_to_local(str(src), str(dst))

    import torch

    @patch_func(torch, 'load')
    def load(f, *args, **kwargs):
        if isinstance(f, str):
            f = io.BytesIO(fileio.get(f))
        return load._fallback(f, *args, **kwargs)

    @patch_func(torch, 'save')
    def save(obj, f, *args, **kwargs):
        backend = get_file_backend(f)
        if isinstance(backend, LocalBackend):
            return save._fallback(obj, f, *args, **kwargs)

        with io.BytesIO() as buffer:
            save._fallback(obj, buffer, *args, **kwargs)
            buffer.seek(0)
            backend.put(buffer, f)

        # from tempfile import TemporaryDirectory
        # import os
        # with TemporaryDirectory(dir='/dev/shm') as tmpdir:
        #     suffix = os.path.split(f)[-1]
        #     tmppath = os.path.join._fallback(tmpdir, suffix)
        #     from mmengine import print_log
        #     print_log('write to tmp dir', logger='current')
        #     save._fallback(obj, tmppath, *args, **kwargs)
        #     print_log('write to ceph', logger='current')

        #     with open(tmppath, 'rb') as buffer:
        #         backend.put(buffer, f)

    from sentencepiece import SentencePieceProcessor

    @patch_func(SentencePieceProcessor, 'LoadFromFile')
    def LoadFromFile(cls, path):
        if path:
            backend = get_file_backend(path)
            if isinstance(backend, LocalBackend):
                return LoadFromFile._fallback(cls, path)
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as tmpdir:
                local_path = backend.copyfile_to_local(path, tmpdir)
                loaded_file = LoadFromFile._fallback(cls, local_path)
            return loaded_file
        else:
            return LoadFromFile._fallback(cls, path)

    try:
        setattr(patch_fileio, '_patched', True)
        yield
    finally:
        for patched_fn in patch_func._backup:
            (module, fn_name_to_wrap, fn_to_wrap) = patched_fn
            setattr(module, fn_name_to_wrap, fn_to_wrap)
        if global_vars is not None and 'open' in global_vars:
            global_vars['open'] = bak_open
        setattr(patch_fileio, '_patched', False)


def patch_hf_auto_from_pretrained(petrel_hub):
    if hasattr(patch_hf_auto_from_pretrained, '_patched'):
        return

    from peft import PeftModel
    from transformers import (AutoConfig, AutoFeatureExtractor,
                              AutoImageProcessor, AutoModelForCausalLM,
                              AutoProcessor, AutoTokenizer,
                              ImageProcessingMixin, PreTrainedModel,
                              PreTrainedTokenizerBase, ProcessorMixin)
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    target_cls = list(_BaseAutoModelClass.__subclasses__())
    target_cls.extend([AutoModelForCausalLM] +
                      AutoModelForCausalLM.__subclasses__())
    target_cls.extend([AutoConfig] + AutoConfig.__subclasses__())
    target_cls.extend([AutoTokenizer] + AutoTokenizer.__subclasses__())
    target_cls.extend([AutoImageProcessor] +
                      AutoImageProcessor.__subclasses__())
    target_cls.extend([AutoFeatureExtractor] +
                      AutoFeatureExtractor.__subclasses__())
    target_cls.extend([AutoProcessor] + AutoProcessor.__subclasses__())
    target_cls.extend([PreTrainedTokenizerBase] +
                      PreTrainedTokenizerBase.__subclasses__())
    target_cls.extend([ImageProcessingMixin] +
                      ImageProcessingMixin.__subclasses__())
    target_cls.extend([PreTrainedModel] + PreTrainedModel.__subclasses__())
    target_cls.extend([ProcessorMixin] + ProcessorMixin.__subclasses__())
    target_cls.extend([PeftModel] + PeftModel.__subclasses__())

    import os

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        with patch_fileio():
            model_path = pretrained_model_name_or_path
            model_path = os.path.join(petrel_hub, model_path)
            obj = cls._from_pretrained(model_path, *args, **kwargs)
        return obj

    for cls in set(target_cls):
        if not hasattr(cls, '_from_pretrained'):
            cls._from_pretrained = cls.from_pretrained
            cls.from_pretrained = from_pretrained

    patch_hf_auto_from_pretrained._patched = True


def patch_hf_save_pretrained():
    if hasattr(patch_hf_save_pretrained, '_patched'):
        return

    import torch
    from peft import PeftModel
    from transformers import (AutoConfig, AutoTokenizer, PreTrainedModel,
                              PreTrainedTokenizerBase)
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    target_cls = []
    target_cls.extend([AutoConfig] + AutoConfig.__subclasses__())
    target_cls.extend([AutoTokenizer] + AutoTokenizer.__subclasses__())
    target_cls.extend([PreTrainedTokenizerBase] +
                      PreTrainedTokenizerBase.__subclasses__())
    target_cls.extend([PreTrainedModel] + PreTrainedModel.__subclasses__())

    target_cls.extend([_BaseAutoModelClass] +
                      _BaseAutoModelClass.__subclasses__())
    target_cls.extend([PeftModel] + PeftModel.__subclasses__())

    def _patch_wrap(method):

        def wrapped_method(self, *args, **kwargs):

            with patch_fileio():
                kwargs['save_function'] = torch.save
                kwargs['safe_serialization'] = False

                obj = method(self, *args, **kwargs)
            return obj

        return wrapped_method

    for cls in set(target_cls):
        if hasattr(cls, 'save_pretrained'):
            cls.save_pretrained = _patch_wrap(cls.save_pretrained)

    patch_hf_save_pretrained._patched = True


def patch_deepspeed_engine():
    if hasattr(patch_deepspeed_engine, '_patched'):
        return

    def _copy_recovery_script(self, save_path):
        import os
        from shutil import copyfile

        from deepspeed.utils import zero_to_fp32
        from mmengine import PetrelBackend, get_file_backend
        script = 'zero_to_fp32.py'

        src = zero_to_fp32.__file__
        dst = os.path.join(save_path, script)

        backend = get_file_backend(save_path)
        if isinstance(backend, PetrelBackend):
            backend.copyfile_from_local(src, dst)
        else:
            copyfile(src, dst)
            self._change_recovery_script_permissions(dst)

    from deepspeed.runtime.engine import DeepSpeedEngine
    DeepSpeedEngine._copy_recovery_script = _copy_recovery_script

    patch_deepspeed_engine._patched = True
