import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from torch.utils.benchmark import Measurement, TaskSpec


@dataclass(frozen=True)
class LatencyStatistics:
    raw_sample_count: int
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    maximum_ms: float
    minimum_ms: float
    mean_ms: float
    spread_ms: float
    iqr_ms: float
    total_measured_ns: int


@dataclass(frozen=True)
class Throughput:
    inferences_per_second: float | None
    items_per_second: float | None


def percentile(sorted_values: Sequence[int], quantile: float) -> float:

    position = (len(sorted_values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def torch_measurement(samples: Sequence[int], num_threads: int = 1) -> Measurement:
    return Measurement(
        number_per_run=1,
        raw_times=[sample / 1_000_000_000 for sample in samples],
        task_spec=TaskSpec(stmt="model(inputs)", setup="", num_threads=num_threads),
    )


def summarize_latencies(sample_latencies_ns: Sequence[int]) -> LatencyStatistics:
    if not sample_latencies_ns:
        raise ValueError("at least one latency sample is required")
    if any(type(sample) is not int or sample < 0 for sample in sample_latencies_ns):
        raise ValueError("latency samples must be nonnegative integer nanoseconds")
    ordered = sorted(sample_latencies_ns)


    measurement = torch_measurement(ordered) if percentile(ordered, 0.5) > 0 else None
    return LatencyStatistics(
        raw_sample_count=len(ordered),
        p50_ms=measurement.median * 1_000 if measurement else 0.0,
        p90_ms=percentile(ordered, 0.90) / 1_000_000,
        p95_ms=percentile(ordered, 0.95) / 1_000_000,
        p99_ms=percentile(ordered, 0.99) / 1_000_000,
        maximum_ms=ordered[-1] / 1_000_000,
        minimum_ms=ordered[0] / 1_000_000,
        mean_ms=measurement.mean * 1_000 if measurement else statistics.mean(ordered) / 1_000_000,
        spread_ms=(ordered[-1] - ordered[0]) / 1_000_000,
        iqr_ms=measurement.iqr * 1_000 if measurement else (percentile(ordered, 0.75) - percentile(ordered, 0.25)) / 1_000_000,
        total_measured_ns=sum(ordered),
    )


def calculate_throughput(sample_latencies_ns: Sequence[int], batch_size: int) -> Throughput:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    summary = summarize_latencies(sample_latencies_ns)
    if summary.total_measured_ns == 0:
        return Throughput(None, None)
    rate = summary.raw_sample_count * 1_000_000_000 / summary.total_measured_ns
    return Throughput(rate, rate * batch_size)
