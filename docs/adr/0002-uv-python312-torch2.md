# uv, Python 3.12, Torch 2.x on one RTX 4080

The author code is Python 3.6 / torch 1.1 / conda. We standardise on **uv** (`uv add` / `uv sync` / `uv run`), CPython 3.12, and Torch 2.x with CUDA, because that is what actually runs on the 4080 and because agents must not invent a second env tool.

**Status:** accepted

**Considered options:** keep conda + `requirements.txt`; Poetry. Rejected: conda is what we are leaving; Poetry duplicates the lock uv already owns.
