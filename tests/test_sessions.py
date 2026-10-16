import os
from functools import partial

import pytest

from fuse import BenchmarkConfig, create_tiny_workload, run_sessions


def failed_factory():
    raise ValueError("intentional factory failure")


def test_sessions_are_fresh_processes_and_keep_their_own_samples():
    report = run_sessions(partial(create_tiny_workload, input_shape=(2, 4), seed=7),
                          BenchmarkConfig(warmup_count=1, repetition_count=3), session_count=2)
    assert len(report.sessions) == 2
    pids = [session.environment.process_id for session in report.sessions]
    assert len(set(pids)) == 2
    assert os.getpid() not in pids
    for session in report.sessions:
        assert session.raw_sample_count == 6
        assert session.workload.input_shape == (2, 4)
        assert session.workload.input_seed == 7
        assert session.environment.torch_num_threads == 1
        assert session.samples[0].sequence == 0
        rates = session.steady_state_throughput
        assert rates.items_per_second == pytest.approx(2 * rates.inferences_per_second)
    assert "few_sessions" in {warning.code for warning in report.warnings}


def test_worker_failure_is_propagated():
    with pytest.raises(RuntimeError, match="intentional factory failure"):
        run_sessions(failed_factory, session_count=1)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_session_count(count):
    with pytest.raises(ValueError):
        run_sessions(session_count=count)
