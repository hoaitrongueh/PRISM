import multiprocessing
import traceback

from .benchmark import WorkloadFactory, run_benchmark
from .schema import BenchmarkConfig, BenchmarkReport, MeasurementWarning
from .workload import create_tiny_workload


def _worker(connection, factory: WorkloadFactory, config: BenchmarkConfig) -> None:
    try:
        connection.send((True, run_benchmark(factory, config)))
    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def run_sessions(
    workload_factory: WorkloadFactory = create_tiny_workload,
    config: BenchmarkConfig = BenchmarkConfig(),
    session_count: int = 3,
) -> BenchmarkReport:


    if type(session_count) is not int or session_count < 1:
        raise ValueError("session_count must be a positive integer")
    context = multiprocessing.get_context("spawn")
    results = []
    for _ in range(session_count):
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=_worker, args=(writer, workload_factory, config))
        try:
            process.start()
            writer.close()
            try:
                success, payload = reader.recv()
            except EOFError as error:
                raise RuntimeError("benchmark worker exited without a result") from error
            process.join()
            if not success:
                raise RuntimeError(f"benchmark worker failed:\n{payload}")
            if process.exitcode != 0:
                raise RuntimeError(f"benchmark worker exited with code {process.exitcode}")
            results.append(payload)
        finally:
            reader.close()
            writer.close()
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join()
                process.close()
    warnings = []
    if session_count < config.minimum_session_count:
        warnings.append(MeasurementWarning("few_sessions", "sessions",
            f"Only {session_count} independent process session(s); this is limited evidence."))
    medians = [result.phase_statistics["steady_state"].p50_ms for result in results]
    if min(medians) > 0 and (max(medians) - min(medians)) / min(medians) >= config.variability_iqr_ratio_threshold:
        warnings.append(MeasurementWarning("session_median_variation", "sessions",
            f"Session p50 values span {min(medians):.6f} to {max(medians):.6f} ms; cause unknown."))
    return BenchmarkReport(tuple(results), tuple(warnings))
