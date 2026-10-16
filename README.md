# fuse

PyTorch CPU inference benchmarks.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m fuse --sessions 3 --output benchmark-result.json
```

Records loading, first inference, warmup, and steady-state samples in JSON.
Sessions run in separate processes. A fixed seed reproduces weights and inputs,
not timing. CPU only for now.
