from fuse import BenchmarkConfig
from fuse.diagnostics import measurement_warnings
from fuse.schema import Sample


def make_samples(durations):
    return tuple(Sample("steady_state", index, index, duration) for index, duration in enumerate(durations))


def test_high_variability_warning_uses_pytorch_measurement_heuristic():
    warnings = measurement_warnings(make_samples([10_000, 20_000, 30_000, 40_000]), BenchmarkConfig(), 1)
    codes = {warning.code for warning in warnings}
    assert {"high_iqr", "few_tail_samples"} <= codes
    assert "cause unknown" in next(w.message for w in warnings if w.code == "high_iqr")


def test_outlier_warning_keeps_the_sample():
    samples = make_samples([10_000] * 99 + [1_000_000])
    warnings = measurement_warnings(samples, BenchmarkConfig(), 1)
    assert [w.code for w in warnings] == ["upper_outliers"]
    assert "1 durations" in warnings[0].message
    assert len(samples) == 100 and samples[-1].duration_ns == 1_000_000


def test_constant_samples_do_not_trigger_variability_warning():
    assert measurement_warnings(make_samples([10_000] * 100), BenchmarkConfig(), 1) == ()


def test_zero_total_and_clock_resolution_warnings():
    warnings = measurement_warnings(make_samples([0, 0]), BenchmarkConfig(0, 2), 100)
    assert {"zero_total_duration", "zero_median", "near_clock_resolution", "no_warmup"} <= {w.code for w in warnings}


def test_variability_threshold_is_configurable():
    warnings = measurement_warnings(make_samples([10_000, 20_000, 30_000]),
                                    BenchmarkConfig(1, 3, variability_iqr_ratio_threshold=1), 1)
    assert "high_iqr" not in {w.code for w in warnings}
