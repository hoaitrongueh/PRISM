import json
from dataclasses import replace

import pytest

from fuse import result_to_dict, result_to_json, write_result_json
from fuse.schema import BenchmarkReport


def test_json_round_trip_retains_all_samples_and_metadata(result, tmp_path):
    report = BenchmarkReport((result,))
    encoded = result_to_json(report)
    decoded = json.loads(encoded)
    assert decoded == result_to_dict(report)
    session = decoded["sessions"][0]
    assert session["raw_sample_count"] == len(session["samples"]) == 9
    assert session["workload"]["input_shape"] == [1, 16]
    assert session["config"]["warmup_count"] == 2
    assert session["phase_statistics"]["steady_state"]["raw_sample_count"] == 5
    assert session["samples"][-1]["duration_ns"] == result.samples[-1].duration_ns
    assert session["measurement"]["clock"] == "time.perf_counter_ns"
    path = tmp_path / "result.json"
    write_result_json(report, path)
    assert path.read_text(encoding="utf-8") == encoded
    assert encoded.endswith("\n")


def test_dict_is_an_independent_copy(result):
    payload = result_to_dict(result)
    payload["samples"][0]["duration_ns"] = -1
    payload["workload"]["input_shape"][0] = -1
    assert result.samples[0].duration_ns >= 0
    assert result.workload.input_shape == (1, 16)


def test_zero_throughput_is_json_null(result):
    samples = tuple(replace(sample, duration_ns=0) for sample in result.samples)
    payload = json.loads(result_to_json(replace(result, samples=samples)))
    assert payload["steady_state_throughput"]["inferences_per_second"] is None
    assert len(payload["samples"]) == 9


def test_non_finite_json_values_are_rejected(result):
    bad = replace(result, measurement=replace(result.measurement, clock_resolution_ns=float("nan")))
    with pytest.raises(ValueError):
        result_to_json(bad)


def test_write_does_not_silently_create_parent_directories(result, tmp_path):
    with pytest.raises(FileNotFoundError):
        write_result_json(result, tmp_path / "missing" / "result.json")
