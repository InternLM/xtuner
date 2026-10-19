import io
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer
from xtuner.v1.datasets.mllm_tokenize_fn import Glm53VLTokenizeFnConfig, OSSLoaderConfig
from xtuner.v1.datasets.mllm_tokenize_fn.glm53_vl_tokenize_fn import materialize_image_plan

@pytest.mark.parametrize('oss', [False, True])
def test_eager_lazy_pixels_and_text(tmp_path, monkeypatch, oss):
    model=os.environ.get('GLM_5_3_FLASH_PATH')
    if not model:
        pytest.skip('GLM_5_3_FLASH_PATH is required for real-processor parity')
    tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=True)
    image=Image.new('RGB',(448,336),(33,77,120)); path=tmp_path/'image.png';image.save(path)
    buf=io.BytesIO();image.save(buf,format='PNG')
    url='test:s3://bucket/image.png' if oss else str(path)
    item={'messages':[{'role':'user','content':[{'type':'image','image':{'url':url,'image_wh':[448,336]}},
        {'type':'text','text':'Describe it.'}]},{'role':'assistant','content':'A colored image.'}]}
    fn=Glm53VLTokenizeFnConfig(processor_path=model,max_pixels=8000).build(tokenizer,anno_name='parity')
    if oss:
        class Client:
            def __init__(self, **kwargs): pass
            def get(self,path):
                assert path==url
                return buf.getvalue()
        monkeypatch.setitem(sys.modules,'petrel_client.client',SimpleNamespace(Client=Client))
        fn.oss_loader=SimpleNamespace(client=Client())
        fn._lazy_oss_cfg={'backend':'petrel','backend_kwargs':{'conf_path':'test-only'}}
    # Runtime API gathers paths and applies the common dirty-data dispatcher.
    monkeypatch.setenv('XTUNER_LAZY_MEDIA','0')
    eager=fn(item)
    monkeypatch.setenv('XTUNER_LAZY_MEDIA','1')
    lazy=fn(item)
    for key in ('input_ids','labels','num_img_tokens'):
        assert eager[key]==lazy[key]
    for threads in ('0','2'):
        monkeypatch.setenv('XTUNER_LAZY_MEDIA_THREADS',threads)
        pixels,grid=materialize_image_plan(lazy['media_plan'])
        assert torch.equal(pixels,eager['pixel_values'])
        assert torch.equal(grid,eager['image_grid_thw'])
        for dtype in (torch.bfloat16, torch.float32):
            typed_plan = dict(lazy['media_plan'], media_dtype=dtype)
            typed_pixels, typed_grid = materialize_image_plan(typed_plan)
            assert typed_pixels.dtype == dtype
            assert torch.equal(typed_pixels, eager['pixel_values'].to(dtype))
            assert torch.equal(typed_grid, eager['image_grid_thw'])
    # A malformed-size eager fallback can share a pack with ordinary lazy images.
    import copy
    from xtuner.v1.datasets.collator import glm53_vl_sft_collator
    mixed = [copy.deepcopy(eager), copy.deepcopy(lazy)]
    reference = [copy.deepcopy(eager), copy.deepcopy(eager)]
    kwargs = dict(pack_max_length=8192, padding_token_idx=0,
                  image_token_id=fn.processor.image_token_id, merge_unit=fn.merge_unit)
    got = glm53_vl_sft_collator([mixed], **kwargs)[0]
    expected = glm53_vl_sft_collator([reference], **kwargs)[0]
    for key in ('input_ids','pixel_values','image_grid_thw','mm_token_type_ids'):
        assert torch.equal(getattr(got['seq_ctx'],key),getattr(expected['seq_ctx'],key))
    assert torch.equal(got['shifted_labels'],expected['shifted_labels'])
    assert got['seq_ctx'].media_plan is None

    for dtype in (torch.bfloat16, torch.float32):
        packed = glm53_vl_sft_collator([[copy.deepcopy(lazy)]], media_dtype=dtype, **kwargs)[0]
        plan = packed['seq_ctx'].media_plan
        assert plan['media_dtype'] == dtype
        actual, grid = materialize_image_plan(plan)
        expected = glm53_vl_sft_collator([[copy.deepcopy(eager)]], media_dtype=dtype, **kwargs)[0]
        assert actual.dtype == dtype
        assert torch.equal(actual, expected['seq_ctx'].pixel_values)
        assert torch.equal(grid, expected['seq_ctx'].image_grid_thw)
