# PRISM

PyTorch CPU inference benchmarks.

Records loading, first inference, warmup, and steady-state samples in JSON.
Sessions run in separate processes. A fixed seed reproduces weights and inputs,
not timing. CPU only for now.
