import pytest
import torch
from torch import nn

from fuse import BenchmarkConfig, TorchCPUWorkload, run_benchmark
from fuse import benchmark


def test_phase_boundaries_inference_context_and_output_lifetime(monkeypatch):
    events = []

    class Output:
        def __del__(self):
            events.append("release")

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Identity().eval()

        def forward(self, inputs):
            assert torch.is_inference_mode_enabled()
            assert not torch.is_grad_enabled()
            assert not self.training and not self.child.training
            events.append("forward")
            return Output()

    model = Probe()
    workload = TorchCPUWorkload(model, torch.ones(1, 2), "probe")
    durations = [9_000, 1_000, 1_100, 1_200, 1_300, 1_400, 50_000]
    ticks = []
    cursor = 0
    for duration in durations:
        ticks.extend([cursor, cursor + duration])
        cursor += duration + 1_000
    iterator = iter(ticks)

    def clock():
        events.append("clock")
        return next(iterator)

    def factory():
        events.append("factory")
        return workload

    monkeypatch.setattr(benchmark.time, "perf_counter_ns", clock)
    threads_before = torch.get_num_threads()
    result = run_benchmark(factory, BenchmarkConfig(warmup_count=2, repetition_count=3))
    assert events == ["clock", "factory", "clock"] + ["clock", "forward", "clock", "release"] * 6
    assert [sample.duration_ns for sample in result.samples] == durations
    assert result.phase_statistics["model_loading"].total_measured_ns == 9_000
    assert result.phase_statistics["first_inference"].total_measured_ns == 1_000
    assert result.phase_statistics["warmup"].total_measured_ns == 2_300
    assert result.phase_statistics["steady_state"].total_measured_ns == 52_700
    assert result.steady_state_throughput.inferences_per_second == pytest.approx(3e9 / 52_700)
    assert model.training and not model.child.training
    assert torch.get_num_threads() == threads_before


def test_failure_restores_threads_and_all_module_modes():
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Identity().eval()

        def forward(self, inputs):
            assert torch.is_inference_mode_enabled()
            raise RuntimeError("forward failed")

    model = Broken()
    workload = TorchCPUWorkload(model, torch.ones(1, 2), "broken")
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="forward failed"):
        run_benchmark(lambda: workload, BenchmarkConfig(0, 1))
    assert model.training and not model.child.training
    assert torch.get_num_threads() == threads
    assert not torch.is_inference_mode_enabled()


def test_factory_failure_restores_threads():
    def factory():
        raise RuntimeError("loading failed")

    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="loading failed"):
        run_benchmark(factory)
    assert torch.get_num_threads() == threads


def test_zero_warmup_is_supported():
    result = run_benchmark(config=BenchmarkConfig(0, 1))
    assert [sample.phase for sample in result.samples] == ["model_loading", "first_inference", "steady_state"]
    assert "warmup" not in result.phase_statistics
    assert "no_warmup" in {warning.code for warning in result.warnings}


def test_workload_placement_is_revalidated_before_execution():
    workload = TorchCPUWorkload(nn.Linear(2, 2), torch.ones(1, 2), "moved-model")
    workload.model.to("meta")
    with pytest.raises(ValueError, match="CPU"):
        run_benchmark(lambda: workload)
