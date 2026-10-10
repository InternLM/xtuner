"""CPU contracts; runnable with unittest without importing XTuner's conftest."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from .glm53_parity_cases import case_contract, load_glm53_reference


class TestGlm53ParityContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, -100, 3, 4]]),
            "positions": torch.tensor([1, 2]),
            "pixel_values": torch.tensor([[0.25, 0.5]], dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 0, 0]]),
        }
        self.cases = [("image", self.batch)]
        self.checkpoint = self.directory / "checkpoint"
        self.metadata = {
            "backend_name": "automodel",
            "model_class": "nemo_automodel.models.Glm5NextForConditionalGeneration",
            "checkpoint": str(self.checkpoint),
        }
        self.write_metadata()
        self.contract = case_contract(*self.cases[0])
        self.result = {key: self.contract[key] for key in ("name", "positions", "targets", "valid_tokens")}
        self.result.update(mean_ce=1.25, logits_file="image.safetensors")
        (self.directory / "manifest.json").write_text(json.dumps({"schema_version": 1, "cases": [self.contract]}))
        save_file({"logits": torch.ones(2, 8)}, self.directory / "image.safetensors")
        self.write_result()

    def write_metadata(self):
        (self.directory / "metadata-rank0.json").write_text(json.dumps(self.metadata))

    def write_result(self):
        (self.directory / "results.json").write_text(json.dumps({"cases": [self.result]}))

    def test_accepts_identical_inputs_and_shifted_targets(self):
        self.assertEqual(self.contract["targets"], [3, 4])
        self.assertEqual(self.contract["valid_tokens"], 2)
        result = load_glm53_reference(self.directory, self.cases, self.checkpoint)
        self.assertEqual(result[0][0], 1.25)
        torch.testing.assert_close(result[0][1], torch.ones(2, 8))

    def test_rejects_preprocessing_or_supervision_changes(self):
        for key in self.batch:
            with self.subTest(key=key):
                batch = copy.deepcopy(self.batch)
                batch[key].reshape(-1)[0] += 1
                with self.assertRaises(ValueError):
                    load_glm53_reference(self.directory, [("image", batch)], self.checkpoint)

    def test_rejects_result_position_target_count_and_order_mismatch(self):
        for key, changed in (("positions", [2, 1]), ("targets", [4, 3]), ("valid_tokens", 3), ("name", "other")):
            with self.subTest(key=key):
                original = self.result[key]
                self.result[key] = changed
                self.write_result()
                with self.assertRaises(ValueError):
                    load_glm53_reference(self.directory, self.cases, self.checkpoint)
                self.result[key] = original

    def test_rejects_nonfinite_loss_and_logits(self):
        self.result["mean_ce"] = float("nan")
        self.write_result()
        with self.assertRaises(ValueError):
            load_glm53_reference(self.directory, self.cases, self.checkpoint)
        self.result["mean_ce"] = 1.25
        self.write_result()
        save_file({"logits": torch.full((2, 8), float("inf"))}, self.directory / "image.safetensors")
        with self.assertRaises(ValueError):
            load_glm53_reference(self.directory, self.cases, self.checkpoint)

    def test_rejects_wrong_logits_shape(self):
        save_file({"logits": torch.ones(1, 8)}, self.directory / "image.safetensors")
        with self.assertRaises(ValueError):
            load_glm53_reference(self.directory, self.cases, self.checkpoint)

    def test_rejects_wrong_backend_model_or_checkpoint(self):
        for key, changed in (
            ("backend_name", "hf"),
            ("model_class", "transformers.Glm5NextForConditionalGeneration"),
            ("model_class", "nemo_automodel.OtherModel"),
            ("checkpoint", str(self.directory / "other-checkpoint")),
        ):
            with self.subTest(key=key, value=changed):
                original = self.metadata[key]
                self.metadata[key] = changed
                self.write_metadata()
                with self.assertRaises(ValueError):
                    load_glm53_reference(self.directory, self.cases, self.checkpoint)
                self.metadata[key] = original

    def test_rejects_path_escape(self):
        self.result["logits_file"] = "../outside.safetensors"
        self.write_result()
        with self.assertRaises(ValueError):
            load_glm53_reference(self.directory, self.cases, self.checkpoint)


if __name__ == "__main__":
    unittest.main()
