from dataclasses import dataclass

import torch
from torch import nn


DTYPES = {"float32": torch.float32, "float64": torch.float64}
MODEL_IDENTIFIER = "fuse.tiny_mlp.v1"


def validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")


def validate_shape(shape: tuple[int, ...]) -> None:
    if not shape or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("input_shape must contain positive integer dimensions")


class TinyMLP(nn.Module):


    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 32), nn.ReLU(), nn.Linear(32, 8)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


@dataclass(frozen=True)
class TorchCPUWorkload:


    model: nn.Module
    inputs: torch.Tensor
    model_identifier: str
    input_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(self.inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if not isinstance(self.model_identifier, str) or not self.model_identifier.strip():
            raise ValueError("model_identifier must be a nonempty string")
        if self.input_seed is not None:
            validate_seed(self.input_seed)
        self.validate()

    def validate(self) -> None:

        validate_shape(tuple(self.inputs.shape))
        tensors = (self.inputs, *self.model.parameters(), *self.model.buffers())
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise ValueError("M1 supports CPU tensors and models only")


def create_tiny_workload(
    input_shape: tuple[int, int] = (1, 16),
    dtype: str = "float32",
    seed: int = 0,
) -> TorchCPUWorkload:


    validate_seed(seed)
    validate_shape(input_shape)
    if len(input_shape) != 2:
        raise ValueError("TinyMLP input_shape must be (batch_size, input_features)")
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {tuple(DTYPES)}")
    with torch.random.fork_rng(devices=[]):

        torch.random.default_generator.manual_seed(seed)
        model = TinyMLP(input_shape[1]).to(device="cpu", dtype=DTYPES[dtype]).eval()
        inputs = torch.randn(input_shape, dtype=DTYPES[dtype], device="cpu")
    return TorchCPUWorkload(model, inputs, MODEL_IDENTIFIER, seed)
