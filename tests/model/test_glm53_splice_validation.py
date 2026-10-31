"""CPU checks for the modality/token contract at the GLM53 splice boundary."""

from unittest.mock import Mock

import pytest
import torch

from xtuner.v1.model.compose.glm53.modeling_glm53 import Glm53ForConditionalGeneration


class TestGlm53SpliceValidation:
    @pytest.mark.parametrize("shape", [(1, 3), (2, 4)])
    @pytest.mark.parametrize("use_sp", [False, True])
    def test_misaligned_modality_ids_raise_before_collectives(
        self, shape: tuple[int, int], use_sp: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gather = Mock(side_effect=AssertionError("Shape validation must precede SP collectives"))
        monkeypatch.setattr("xtuner.v1.model.compose.glm53.modeling_glm53.dist.all_gather", gather)
        mesh = Mock() if use_sp else None
        if mesh is not None:
            mesh.size.return_value = 2

        with pytest.raises(ValueError, match="pack lost modality alignment"):
            Glm53ForConditionalGeneration._splice(
                None, torch.zeros(1, 4, 2), torch.zeros(shape, dtype=torch.long), torch.zeros(1, 2), {1: 1}, mesh
            )
        gather.assert_not_called()

    def test_aligned_splice_preserves_text_and_places_visual_features(self) -> None:
        embeds = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
        original = embeds.clone()
        features = torch.tensor([[10.0, 11.0], [20.0, 21.0]], requires_grad=True)
        ids = torch.tensor([[0, 1, 0, 2]])

        result = Glm53ForConditionalGeneration._splice(None, embeds, ids, features, {1: 1, 2: 1})

        torch.testing.assert_close(result[ids == 0], original[ids == 0])
        torch.testing.assert_close(result[ids == 1], features[:1])
        torch.testing.assert_close(result[ids == 2], features[1:])
        result.sum().backward()
        torch.testing.assert_close(features.grad, torch.ones_like(features))
