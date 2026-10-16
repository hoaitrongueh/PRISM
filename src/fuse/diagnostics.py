from .schema import BenchmarkConfig, MeasurementWarning, Sample
from .statistics import percentile, torch_measurement


def measurement_warnings(
    samples: tuple[Sample, ...], config: BenchmarkConfig, clock_resolution_ns: float
) -> tuple[MeasurementWarning, ...]:
    warnings = []
    steady = [sample.duration_ns for sample in samples if sample.phase == "steady_state"]
    if len(steady) < config.minimum_tail_samples:
        warnings.append(MeasurementWarning("few_tail_samples", "steady_state",
            f"Only {len(steady)} samples; p95/p99 are descriptive estimates from a small sample."))
    if config.warmup_count == 0:
        warnings.append(MeasurementWarning("no_warmup", "warmup", "No warmup calls were requested after first inference."))
    for phase in ("model_loading", "first_inference", "warmup", "steady_state"):
        values = [sample.duration_ns for sample in samples if sample.phase == phase]
        count = sum(value <= config.clock_resolution_warning_multiplier * clock_resolution_ns for value in values)
        if count:
            warnings.append(MeasurementWarning("near_clock_resolution", phase,
                f"{count} durations are <= {config.clock_resolution_warning_multiplier:g} times nominal clock resolution; all are retained. Clock-read overhead is not subtracted."))
    if steady:
        ordered = sorted(steady)
        measurement = torch_measurement(steady, config.num_threads) if percentile(ordered, 0.5) > 0 else None
        if measurement and not measurement.meets_confidence(config.variability_iqr_ratio_threshold):
            ratio = measurement.iqr / measurement.median
            warnings.append(MeasurementWarning("high_iqr", "steady_state",
                f"IQR/median is {ratio:.3f}, at or above threshold {config.variability_iqr_ratio_threshold:.3f}; cause unknown."))
        if measurement is None:
            warnings.append(MeasurementWarning("zero_median", "steady_state",
                "Median duration is zero; the IQR/median variability heuristic is undefined."))
        q25, q75 = percentile(ordered, 0.25), percentile(ordered, 0.75)
        fence = q75 + config.outlier_iqr_multiplier * (q75 - q25)
        count = sum(value > fence for value in steady)
        if count:
            warnings.append(MeasurementWarning("upper_outliers", "steady_state",
                f"{count} durations exceed p75 + {config.outlier_iqr_multiplier:g} * IQR; retained without attributing a cause."))
        if sum(steady) == 0:
            warnings.append(MeasurementWarning("zero_total_duration", "steady_state",
                "Total duration is zero; throughput is null."))
    return tuple(warnings)
