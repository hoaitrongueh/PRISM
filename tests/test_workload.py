import pytest
import torch
from torch import nn

from fuse import TorchCPUWorkload, create_tiny_workload


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_seed_reproduces_parameters_inputs_and_outputs_without_rng_mutation(dtype):
    state_before = torch.random.get_rng_state().clone()
    first = create_tiny_workload((2, 7), dtype, 42)
    second = create_tiny_workload((2, 7), dtype, 42)
    assert torch.equal(state_before, torch.random.get_rng_state())
    assert torch.equal(first.inputs, second.inputs)
    for left, right in zip(first.model.parameters(), second.model.parameters()):
        assert torch.equal(left, right)
    with torch.inference_mode():
        assert torch.equal(first.model(first.inputs), second.model(second.inputs))
        assert first.model(first.inputs).shape == (2, 8)
    assert first.inputs.dtype == getattr(torch, dtype)
    assert first.inputs.device.type == "cpu"
    assert not first.model.training
    assert not torch.equal(first.inputs, create_tiny_workload((2, 7), dtype, 43).inputs)


@pytest.mark.parametrize("kwargs", [
    {"input_shape": ()}, {"input_shape": (1,)}, {"input_shape": (1, 2, 3)},
    {"input_shape": (0, 16)}, {"input_shape": (True, 16)}, {"input_shape": (1, 1.5)},
    {"dtype": "float16"}, {"seed": -1}, {"seed": 2**63}, {"seed": True},
])
def test_invalid_workload_configuration(kwargs):
    with pytest.raises(ValueError):
        create_tiny_workload(**kwargs)


def test_non_cpu_input_is_rejected_without_requiring_cuda():
    with pytest.raises(ValueError, match="CPU"):
        TorchCPUWorkload(nn.Identity(), torch.empty((1, 2), device="meta"), "meta-input")


def test_non_cpu_parameters_and_buffers_are_rejected():
    with pytest.raises(ValueError, match="CPU"):
        TorchCPUWorkload(nn.Linear(2, 2, device="meta"), torch.ones(1, 2), "meta-model")
    model = nn.Identity()
    model.register_buffer("bad", torch.empty(1, device="meta"))
    with pytest.raises(ValueError, match="CPU"):
        TorchCPUWorkload(model, torch.ones(1, 2), "meta-buffer")


def test_invalid_custom_workload():
    with pytest.raises(TypeError, match="Module"):
        TorchCPUWorkload(object(), torch.ones(1, 2), "bad")
    with pytest.raises(TypeError, match="Tensor"):
        TorchCPUWorkload(nn.Identity(), [1], "bad")
    with pytest.raises(ValueError, match="model_identifier"):
        TorchCPUWorkload(nn.Identity(), torch.ones(1, 2), " ")
