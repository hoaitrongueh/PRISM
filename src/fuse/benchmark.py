import os
import platform
import time
from collections.abc import Callable

import torch

from .diagnostics import measurement_warnings
from .schema import (BenchmarkConfig, BenchmarkResult, EnvironmentInfo,
                     MeasurementInfo, Phase, Sample, WorkloadInfo)
from .workload import TorchCPUWorkload, create_tiny_workload

WorkloadFactory = Callable[[], TorchCPUWorkload]


def measure_calls(workload: TorchCPUWorkload, count: int, phase: Phase, start_sequence: int) -> tuple[Sample, ...]:

    if type(count) is not int or count < 0:
        raise ValueError("call count must be a nonnegative integer")
    samples = []
    model, inputs = workload.model, workload.inputs
    for index in range(count):
        start_ns = time.perf_counter_ns()
        output = model(inputs)
        stop_ns = time.perf_counter_ns()
        del output
        samples.append(Sample(phase, start_sequence + index, index, stop_ns - start_ns))
    return tuple(samples)


def warmup(workload: TorchCPUWorkload, count: int, start_sequence: int) -> tuple[Sample, ...]:

    return measure_calls(workload, count, "warmup", start_sequence)


def _environment() -> EnvironmentInfo:
    return EnvironmentInfo(
        python_version=platform.python_version(), pytorch_version=str(torch.__version__),
        operating_system=platform.platform(),
        cpu_identifier=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "unknown",
        cpu_architecture=platform.machine(), logical_cpu_count=os.cpu_count(), device="cpu",
        process_id=os.getpid(), torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        thread_environment={name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    )


def run_benchmark(
    workload_factory: WorkloadFactory = create_tiny_workload,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> BenchmarkResult:


    original_threads = torch.get_num_threads()
    training_flags = []
    try:
        torch.set_num_threads(config.num_threads)
        environment = _environment()
        clock = time.get_clock_info("perf_counter")
        measurement = MeasurementInfo(clock.resolution * 1_000_000_000, clock.implementation)
        start_ns = time.perf_counter_ns()
        workload = workload_factory()
        stop_ns = time.perf_counter_ns()
        samples = (Sample("model_loading", 0, 0, stop_ns - start_ns),)
        workload.validate()
        workload_info = WorkloadInfo(workload.model_identifier, tuple(workload.inputs.shape),
                                     str(workload.inputs.dtype).removeprefix("torch."), workload.input_seed)
        training_flags = [(module, module.training) for module in workload.model.modules()]
        workload.model.eval()
        with torch.inference_mode():
            samples += measure_calls(workload, 1, "first_inference", len(samples))
            samples += warmup(workload, config.warmup_count, len(samples))
            samples += measure_calls(workload, config.repetition_count, "steady_state", len(samples))
        warnings = measurement_warnings(samples, config, measurement.clock_resolution_ns)
        return BenchmarkResult(environment, workload_info, config, measurement, samples, warnings)
    finally:
        for module, training in training_flags:
            module.training = training
        torch.set_num_threads(original_threads)
