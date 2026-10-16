import pytest

from fuse import BenchmarkConfig, run_benchmark


@pytest.fixture(scope="session")
def result():
    return run_benchmark(config=BenchmarkConfig(warmup_count=2, repetition_count=5))
