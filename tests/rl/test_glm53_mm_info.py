"""CPU regression for GLM modality metadata, packing and error provenance."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from pydantic import ValidationError

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.vlm_jsonl import VLMJsonlDataset
from xtuner.v1.rl.trainer.train_batch import TrainBatch, get_train_seq_ctx


class TestGLMModality(unittest.TestCase):
    def sample(self, mm_info):
        state = RolloutState(
            message=[],
            input_ids=[10, 11, 12, 20, 21],
            labels=[-100, -100, -100, 20, 21],
            mm_info=mm_info,
            reward={"score": 1},
        )
        adapter = SimpleNamespace(rollout_teacher_targets=lambda *args, **kwargs: None)
        return TrainBatch._build_sample(None, state, False, adapter)

    def test_prompt_modality_and_generated_text(self):
        sample = self.sample({"mm_token_type_ids": np.array([[0, 1, 1]]), "num_img_tokens": [2]})
        self.assertEqual(sample.seq_ctx.mm_token_type_ids.tolist(), [[0, 1, 1, 0]])
        self.assertEqual(sample.seq_ctx.input_ids.shape, sample.seq_ctx.mm_token_type_ids.shape)
        self.assertEqual(sample.shifted_labels.tolist(), [[-100, -100, 20, 21]])

    def test_text_and_optional_legacy_payload(self):
        for info in (None, {"num_img_tokens": []}):
            self.assertIsNone(self.sample(info).seq_ctx.mm_token_type_ids)
        self.assertEqual(
            self.sample({"mm_token_type_ids": np.zeros((1, 3), dtype=np.int64)}).seq_ctx.mm_token_type_ids.tolist(),
            [[0, 0, 0, 0]],
        )

    def test_pack_text_first_and_image_first(self):
        text = self.sample(None).seq_ctx
        image = self.sample({"mm_token_type_ids": np.array([[0, 1, 1]]), "num_img_tokens": [2]}).seq_ctx
        for contexts, expected in [
            ([text, image], [[0, 0, 0, 0, 0, 1, 1, 0]]),
            ([image, text], [[0, 1, 1, 0, 0, 0, 0, 0]]),
        ]:
            packed = SequenceContext.cat(contexts)
            self.assertEqual(packed.mm_token_type_ids.tolist(), expected)
            self.assertEqual(packed.input_ids.shape, packed.mm_token_type_ids.shape)

    def test_overlong_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be longer"):
            get_train_seq_ctx(
                torch.tensor([[1, 2]]), multimodal_train_info={"mm_token_type_ids": np.array([[0, 1, 1]])}
            )

    def test_unknown_fields_still_rejected(self):
        with self.assertRaises(ValidationError):
            RolloutState(message=[], mm_info={"not_a_real_field": np.array([0])})

    def dataset(self, tokenizer):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "data.jsonl"
        path.write_text(json.dumps({"prompt": []}) + "\n")
        ds = object.__new__(VLMJsonlDataset)
        ds.path = str(path)
        ds._meta = {"offsets": [0]}
        ds.media_root = ""
        ds.tokenize_fn = tokenizer
        ds.fake_data = {"messages": []}
        return ds

    def test_rl_error_is_not_masked_by_fake_prompt(self):
        error = ValueError("original invalid multimodal payload")

        class Tokenizer:
            allow_fake_data_fallback = False

            def __call__(self, data, **kwargs):
                if "prompt" not in data:
                    raise KeyError("prompt")
                raise error

        with self.assertRaises(ValueError) as caught:
            self.dataset(Tokenizer())[0]
        self.assertIs(caught.exception, error)

    def test_default_sft_fallback_unchanged(self):
        def tokenizer(data, **kwargs):
            if "prompt" in data:
                raise ValueError("bad sample")
            return {"input_ids": [1, 2], "labels": [1, 2]}

        self.assertEqual(self.dataset(tokenizer)[0]["labels"], [-100, -100])


if __name__ == "__main__":
    unittest.main()
