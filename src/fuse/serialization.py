import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import BenchmarkReport, BenchmarkResult


def result_to_dict(result: BenchmarkResult | BenchmarkReport) -> dict[str, Any]:
    def arrays(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: arrays(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [arrays(item) for item in value]
        return value
    return arrays(asdict(result))


def result_to_json(result: BenchmarkResult | BenchmarkReport) -> str:
    return json.dumps(result_to_dict(result), indent=2, allow_nan=False) + "\n"


def write_result_json(result: BenchmarkResult | BenchmarkReport, path: str | Path) -> None:
    Path(path).write_text(result_to_json(result), encoding="utf-8")
