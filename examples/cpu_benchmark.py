from fuse import BenchmarkConfig, run_sessions, write_result_json


def main() -> None:
    result = run_sessions(config=BenchmarkConfig(warmup_count=10, repetition_count=100), session_count=3)
    write_result_json(result, "benchmark-result.json")
    print("Wrote benchmark-result.json")


if __name__ == "__main__":
    main()
