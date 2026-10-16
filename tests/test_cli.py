import json

import pytest

from fuse import cli
from fuse.schema import BenchmarkReport


def test_cli_stdout_is_json(monkeypatch, capsys, result):
    def sessions(factory, config, count):
        assert count == 3
        assert config.num_threads == 2
        assert factory.keywords == {}
        assert factory.args == ((1, 16), "float32", 7)
        return BenchmarkReport((result,))

    monkeypatch.setattr(cli, "run_sessions", sessions)
    assert cli.main(["--threads", "2", "--seed", "7"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "1.0"


def test_cli_writes_file_without_mixing_text_into_json(monkeypatch, tmp_path, capsys, result):
    monkeypatch.setattr(cli, "run_sessions", lambda *args: BenchmarkReport((result,)))
    path = tmp_path / "result.json"
    assert cli.main(["--output", str(path)]) == 0
    assert capsys.readouterr().out == ""
    assert json.loads(path.read_text())["sessions"][0]["raw_sample_count"] == 9


@pytest.mark.parametrize("arguments", [
    ["--warmup", "-1"], ["--repetitions", "0"], ["--sessions", "0"],
    ["--threads", "0"], ["--batch-size", "0"], ["--input-features", "-1"],
    ["--seed", "-1"], ["--dtype", "int64"],
])
def test_cli_invalid_arguments(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
