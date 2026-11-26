from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock

from xtuner.v1.model.compose.base import BaseComposeModel


class TestComposeHFLoading(TestCase):
    def test_materialize_prepared_hf_forwards_device_to_every_submodel(self) -> None:
        compose = SimpleNamespace(
            language_model=MagicMock(),
            vision_tower=MagicMock(),
            multi_modal_projector=MagicMock(),
        )
        prepared = {
            "language_model": object(),
            "vision_tower": object(),
            "multi_modal_projector": object(),
        }

        BaseComposeModel.materialize_prepared_hf(compose, prepared, device="cpu")  # type: ignore[arg-type]

        compose.language_model.materialize_prepared_hf.assert_called_once_with(
            prepared["language_model"], device="cpu"
        )
        compose.vision_tower.materialize_prepared_hf.assert_called_once_with(prepared["vision_tower"], device="cpu")
        compose.multi_modal_projector.materialize_prepared_hf.assert_called_once_with(
            prepared["multi_modal_projector"], device="cpu"
        )
