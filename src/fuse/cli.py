import argparse
from functools import partial
from pathlib import Path

from .sessions import run_sessions
from .schema import BenchmarkConfig
from .serialization import result_to_json, write_result_json
from .workload import DTYPES, create_tiny_workload, validate_seed, validate_shape


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CPU inference benchmarks.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-features", type=int, default=16)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        config = BenchmarkConfig(args.warmup, args.repetitions, args.threads)
        if args.sessions < 1:
            raise ValueError("sessions must be a positive integer")
        validate_seed(args.seed)
        validate_shape((args.batch_size, args.input_features))
        factory = partial(create_tiny_workload, (args.batch_size, args.input_features), args.dtype, args.seed)
    except ValueError as error:
        parser.error(str(error))
    result = run_sessions(factory, config, args.sessions)
    if args.output is None:
        print(result_to_json(result), end="")
    else:
        write_result_json(result, args.output)
    return 0
