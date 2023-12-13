import io
from contextlib import contextmanager

import mmengine.fileio as fileio
from mmengine.fileio import LocalBackend, get_file_backend


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
        backend = get_file_backend(a)
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

    import filecmp

    @patch_func(filecmp, 'cmp')
    def cmp(f1, f2, *args, **kwargs):
        with fileio.get_local_path(f1) as f1, fileio.get_local_path(f2) as f2:
            return cmp._fallback(f1, f2, *args, **kwargs)

    import shutil

    @patch_func(shutil, 'copy')
    def copy(src, dst, **kwargs):
        backend = get_file_backend(src)
        if isinstance(backend, LocalBackend):
            return copy._fallback(src, dst, **kwargs)
        return backend.copyfile_to_local(str(src), str(dst))

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

        assert isinstance(f, str)
        with io.BytesIO() as _f:
            save._fallback(obj, _f, *args, **kwargs)
            backend.put(_f.getvalue(), f)

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


def patch_hf_auto_from_pretrained():
    if hasattr(patch_hf_auto_from_pretrained, '_patched'):
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ori_auto_model_method = AutoModelForCausalLM.from_pretrained
    ori_auto_tok_method = AutoTokenizer.from_pretrained

    AutoModelForCausalLM._from_pretrained = ori_auto_model_method
    AutoTokenizer._from_pretrained = ori_auto_tok_method

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        with patch_fileio():
            cls._from_pretrained(*args, **kwargs)

    AutoModelForCausalLM.from_pretrained = from_pretrained
    AutoTokenizer.from_pretrained = from_pretrained

    patch_hf_auto_from_pretrained._patched = True
