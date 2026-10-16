from dataclasses import FrozenInstanceError, replace

import pytest

from fuse import BenchmarkConfig
from fuse.schema import BenchmarkReport, Sample, WorkloadInfo


def test_result_schema_and_phase_counts(result):
    assert result.schema_version == "1.0"
    assert result.backend == "pytorch_cpu"
    assert result.raw_sample_count == 9
    assert [sample.sequence for sample in result.samples] == list(range(9))
    assert [sample.phase for sample in result.samples] == [
        "model_loading", "first_inference", "warmup", "warmup", *(["steady_state"] * 5)
    ]
    assert all(type(sample.duration_ns) is int and sample.duration_ns >= 0 for sample in result.samples)
    assert result.phase_statistics["warmup"].raw_sample_count == 2
    assert result.phase_statistics["steady_state"].raw_sample_count == 5
    assert result.workload.model_identifier == "fuse.tiny_mlp.v1"
    assert result.workload.input_shape == (1, 16)
    assert result.workload.dtype == "float32"
    assert result.workload.input_seed == 0
    assert result.environment.python_version
    assert result.environment.pytorch_version
    assert result.environment.cpu_identifier
    assert result.environment.device == "cpu"
    assert result.environment.torch_num_threads == result.config.num_threads == 1
    assert result.environment.torch_num_interop_threads >= 1
    assert result.measurement.clock == "time.perf_counter_ns"
    assert result.measurement.inference_mode is True
    assert result.measurement.clock_resolution_ns > 0
    assert "OMP_NUM_THREADS" in result.environment.thread_environment


def test_sample_count_and_order_are_validated(result):
    with pytest.raises(ValueError, match="counts"):
        replace(result, samples=result.samples[:-1])
    changed = (replace(result.samples[0], sequence=1), *result.samples[1:])
    with pytest.raises(ValueError, match="sequence"):
        replace(result, samples=changed)
    changed = (*result.samples[:3], replace(result.samples[3], phase_sequence=0), *result.samples[4:])
    with pytest.raises(ValueError, match="sequence"):
        replace(result, samples=changed)


def test_summary_is_recomputed_from_samples(result):
    changed = (*result.samples[:-1], replace(result.samples[-1], duration_ns=10**12))
    updated = replace(result, samples=changed)
    assert updated.phase_statistics["steady_state"].maximum_ms == 1_000_000
    assert updated.raw_sample_count == result.raw_sample_count
    with pytest.raises(FrozenInstanceError):
        updated.samples = ()


@pytest.mark.parametrize("field,value", [
    ("warmup_count", -1), ("warmup_count", True), ("warmup_count", 0.5),
    ("repetition_count", 0), ("repetition_count", False), ("repetition_count", 1.5),
    ("num_threads", 0), ("num_threads", True), ("minimum_tail_samples", 0),
    ("variability_iqr_ratio_threshold", 0), ("variability_iqr_ratio_threshold", float("nan")),
    ("outlier_iqr_multiplier", float("inf")), ("outlier_iqr_multiplier", True),
    ("minimum_session_count", 0), ("clock_resolution_warning_multiplier", -1),
])
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        BenchmarkConfig(**{field: value})


@pytest.mark.parametrize("arguments", [
    ("unknown", 0, 0, 1), ("warmup", -1, 0, 1), ("warmup", 0, -1, 1),
    ("warmup", 0, 0, -1), ("warmup", 0, 0, True),
])
def test_invalid_sample(arguments):
    with pytest.raises(ValueError):
        Sample(*arguments)


def test_empty_report_and_invalid_workload_info():
    with pytest.raises(ValueError):
        BenchmarkReport(())
    with pytest.raises(ValueError):
        WorkloadInfo("", (1, 16), "float32", 0)
