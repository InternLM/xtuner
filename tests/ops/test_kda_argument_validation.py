"""Unsupported KDA options must fail before a kernel is called."""

import pytest
import torch

from xtuner.v1.ops.kda.causal_conv1d import CausalConv1dFunction, causal_conv1d
from xtuner.v1.ops.kda.chunk_kda import ChunkKDAFunction, chunk_kda


@pytest.fixture(params=["conv", "chunk"])
def entrypoint(request, monkeypatch):
    x = torch.zeros(1, 2, 1, 1)
    result = (x, None)
    if request.param == "conv":
        monkeypatch.setattr(CausalConv1dFunction, "apply", lambda *args: x)
        return causal_conv1d, {"x": x.squeeze(-1), "weight": torch.zeros(1, 1)}, result
    monkeypatch.setattr(ChunkKDAFunction, "apply", lambda *args: result)
    return chunk_kda, {"q": x, "k": x, "v": x, "g": x, "beta": x.squeeze(-1)}, result


@pytest.mark.parametrize("value", [torch.zeros(2), torch.tensor(0.0), torch.tensor(1.0)])
def test_unsupported_tensor_options_raise_named_error(entrypoint, value):
    function, inputs, _ = entrypoint
    options = ("residual", "initial_state") if function is causal_conv1d else ("initial_state", "A_log", "dt_bias")
    for option in options:
        with pytest.raises(NotImplementedError, match=option):
            function(**inputs, **{option: value})


@pytest.mark.parametrize("value", [True, torch.tensor(False), torch.tensor(True), torch.zeros(2)])
def test_unsupported_final_state_raises_named_error(entrypoint, value):
    function, inputs, _ = entrypoint
    with pytest.raises(NotImplementedError, match="output_final_state"):
        function(**inputs, output_final_state=value)


def test_non_null_cp_context_is_rejected(entrypoint):
    function, inputs, _ = entrypoint
    with pytest.raises(NotImplementedError, match="cp_context"):
        function(**inputs, cp_context=object())


@pytest.mark.parametrize("final_state", [None, False])
def test_null_options_and_false_final_state_keep_default_behavior(entrypoint, final_state):
    function, inputs, expected = entrypoint
    options = ("residual", "initial_state") if function is causal_conv1d else ("initial_state", "A_log", "dt_bias")
    output, state = function(**inputs, **dict.fromkeys(options), cp_context=None, output_final_state=final_state)
    assert output is expected[0]
    assert state is None
