import pytest

from fuse.statistics import calculate_throughput, summarize_latencies, torch_measurement


def test_known_statistics_and_units():
    samples = [5_000_000, 1_000_000, 4_000_000, 2_000_000, 3_000_000]
    summary = summarize_latencies(samples)
    assert summary.raw_sample_count == 5
    assert summary.p50_ms == pytest.approx(3)
    assert summary.p90_ms == pytest.approx(4.6)
    assert summary.p95_ms == pytest.approx(4.8)
    assert summary.p99_ms == pytest.approx(4.96)
    assert summary.mean_ms == pytest.approx(3)
    assert summary.minimum_ms == 1
    assert summary.maximum_ms == 5
    assert summary.spread_ms == 4
    assert summary.iqr_ms == pytest.approx(2)
    assert summary.total_measured_ns == 15_000_000
    assert samples[0] == 5_000_000
    rates = calculate_throughput(samples, batch_size=4)
    assert rates.inferences_per_second == pytest.approx(5 / 0.015)
    assert rates.items_per_second == pytest.approx(20 / 0.015)


@pytest.mark.parametrize("samples", [[7], [7, 7], [7, 7, 7]])
def test_single_and_constant_samples(samples):
    summary = summarize_latencies(samples)
    assert summary.p50_ms == pytest.approx(7e-6)
    assert summary.p90_ms == summary.p95_ms == summary.p99_ms == 7e-6
    assert summary.spread_ms == 0
    assert summary.iqr_ms == 0


def test_extreme_outlier_is_retained_in_tail_and_throughput():
    samples = [1_000_000] * 99 + [1_000_000_000]
    summary = summarize_latencies(samples)
    assert summary.raw_sample_count == 100
    assert summary.maximum_ms == 1000
    assert summary.mean_ms == pytest.approx(10.99)
    assert calculate_throughput(samples, 1).inferences_per_second == pytest.approx(100 / 1.099)


def test_zero_samples_are_preserved_and_zero_total_has_no_throughput():
    summary = summarize_latencies([0, 0, 0])
    assert summary.raw_sample_count == 3
    assert summary.p50_ms == summary.iqr_ms == summary.mean_ms == 0
    assert calculate_throughput([0, 0], 1).inferences_per_second is None
    assert calculate_throughput([0, 0, 6], 2).items_per_second == pytest.approx(1e9)
    assert summarize_latencies([0, 0, 6]).mean_ms == pytest.approx(2e-6)


def test_measurement_has_one_call_per_observation():
    measurement = torch_measurement([1_000_000, 2_000_000], num_threads=2)
    assert measurement.number_per_run == 1
    assert measurement.raw_times == [0.001, 0.002]
    assert measurement.task_spec.num_threads == 2


@pytest.mark.parametrize("samples", [[], [-1], [1.5], [True], [float("nan")], [float("inf")]])
def test_invalid_samples(samples):
    with pytest.raises(ValueError):
        summarize_latencies(samples)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError):
        calculate_throughput([100], batch_size)
