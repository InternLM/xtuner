"""The logits check must catch errors that cross entropy cannot detect."""

import pytest
import torch
import torch.nn.functional as F

from xtuner._testing.logits import check_logits


def test_identical_logits():
    x = torch.arange(40).reshape(4, 10).float()
    assert check_logits(x, x)["relative_l2"] == 0


def test_common_offset_preserves_ce_but_fails_logits():
    x = torch.arange(40).reshape(4, 10).float() / 10
    y = torch.tensor([0, 1, 2, 3])
    torch.testing.assert_close(F.cross_entropy(x, y), F.cross_entropy(x + 10, y))
    with pytest.raises(AssertionError):
        check_logits(x + 10, x)


def test_position_permutation_is_rejected():
    x = torch.eye(8)
    with pytest.raises(AssertionError):
        check_logits(x.roll(1, 0), x)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_logits_are_rejected(value):
    with pytest.raises(AssertionError):
        check_logits(torch.full((2, 4), value), torch.ones(2, 4))


def test_wrong_shape_is_rejected():
    with pytest.raises(AssertionError):
        check_logits(torch.ones(2, 4), torch.ones(3, 4))
