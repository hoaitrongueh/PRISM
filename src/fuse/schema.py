import math
from dataclasses import dataclass, field
from typing import Literal

from .statistics import LatencyStatistics, Throughput, calculate_throughput, summarize_latencies
from .workload import validate_seed, validate_shape

PHASES = ("model_loading", "first_inference", "warmup", "steady_state")
Phase = Literal["model_loading", "first_inference", "warmup", "steady_state"]


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_count: int = 10
    repetition_count: int = 100
    num_threads: int = 1
    variability_iqr_ratio_threshold: float = 0.10
    outlier_iqr_multiplier: float = 1.5
    minimum_tail_samples: int = 100
    minimum_session_count: int = 3
    clock_resolution_warning_multiplier: float = 10.0

    def __post_init__(self) -> None:
        for name, minimum in (("warmup_count", 0), ("repetition_count", 1),
                              ("num_threads", 1), ("minimum_tail_samples", 1),
                              ("minimum_session_count", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("variability_iqr_ratio_threshold", "outlier_iqr_multiplier", "clock_resolution_warning_multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Sample:
    phase: Phase
    sequence: int
    phase_sequence: int
    duration_ns: int

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"unknown phase: {self.phase}")
        for name in ("sequence", "phase_sequence", "duration_ns"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class EnvironmentInfo:
    python_version: str
    pytorch_version: str
    operating_system: str
    cpu_identifier: str
    cpu_architecture: str
    logical_cpu_count: int | None
    device: str
    process_id: int
    torch_num_threads: int
    torch_num_interop_threads: int
    thread_environment: dict[str, str | None]


@dataclass(frozen=True)
class WorkloadInfo:
    model_identifier: str
    input_shape: tuple[int, ...]
    dtype: str
    input_seed: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        validate_shape(self.input_shape)
        if not self.model_identifier.strip():
            raise ValueError("model_identifier must be nonempty")
        if self.input_seed is not None:
            validate_seed(self.input_seed)


@dataclass(frozen=True)
class MeasurementInfo:
    clock_resolution_ns: float
    clock_implementation: str
    clock: str = "time.perf_counter_ns"
    model_loading_boundary: str = "workload_factory(): model construction, weights, and input creation"
    inference_boundary: str = "one synchronous CPU model(inputs) call, including output allocation"
    excluded_from_inference: str = "setup, mode changes, loop bookkeeping, output destruction, statistics, JSON, process startup"
    first_inference_definition: str = "first call on the newly constructed model; not a cold OS cache guarantee"
    inference_mode: bool = True
    model_mode: str = "eval"
    percentile_method: str = "linear interpolation at (n - 1) * quantile; p50 is median"
    spread_definition: str = "maximum - minimum; IQR = p75 - p25"
    throughput_basis: str = "steady-state calls / sum(steady-state durations); items = calls * batch_size"
    statistics_engine: str = "torch.utils.benchmark.Measurement for p50, mean, IQR and IQR/median confidence heuristic"
    timing_strategy: str = "one explicit clock pair per call; no block averaging or automatic stopping"


@dataclass(frozen=True)
class MeasurementWarning:
    code: str
    phase: str
    message: str


@dataclass(frozen=True)
class BenchmarkResult:
    environment: EnvironmentInfo
    workload: WorkloadInfo
    config: BenchmarkConfig
    measurement: MeasurementInfo
    samples: tuple[Sample, ...]
    warnings: tuple[MeasurementWarning, ...] = ()
    phase_statistics: dict[str, LatencyStatistics] = field(init=False)
    raw_sample_count: int = field(init=False)
    steady_state_throughput: Throughput = field(init=False)
    schema_version: str = field(default="1.0", init=False)
    backend: str = field(default="pytorch_cpu", init=False)

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        expected = (["model_loading", "first_inference"] + ["warmup"] * self.config.warmup_count
                    + ["steady_state"] * self.config.repetition_count)
        if [sample.phase for sample in samples] != expected:
            raise ValueError("samples must match ordered loading/first/warmup/steady-state counts")
        grouped: dict[str, list[int]] = {}
        for sequence, sample in enumerate(samples):
            phase_samples = grouped.setdefault(sample.phase, [])
            if sample.sequence != sequence or sample.phase_sequence != len(phase_samples):
                raise ValueError("sample sequence numbers must be contiguous and zero-based")
            phase_samples.append(sample.duration_ns)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "raw_sample_count", len(samples))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "phase_statistics", {
            phase: summarize_latencies(durations) for phase, durations in grouped.items()
        })
        object.__setattr__(self, "steady_state_throughput", calculate_throughput(
            grouped["steady_state"], self.workload.input_shape[0]
        ))


@dataclass(frozen=True)
class BenchmarkReport:
    sessions: tuple[BenchmarkResult, ...]
    warnings: tuple[MeasurementWarning, ...] = ()
    schema_version: str = field(default="1.0", init=False)
    session_isolation: str = field(default="one fresh spawned process per session, executed sequentially", init=False)

    def __post_init__(self) -> None:
        if not self.sessions:
            raise ValueError("at least one session is required")
        object.__setattr__(self, "sessions", tuple(self.sessions))
        object.__setattr__(self, "warnings", tuple(self.warnings))
