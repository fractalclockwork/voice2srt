# Contributing

## Setup

```bash
git clone https://github.com/fractalclockwork/voice2srt.git
cd voice2srt
uv venv --python 3.11
uv sync --all-groups
```

## Tests

Fast tests (no GPU, no large reference audio):

```bash
uv run pytest -q
```

Optional **integration** test (full Whisper pipeline vs a golden SRT) requires CUDA, local files under `data/`, and:

```bash
VOICE2SRT_E2E=1 uv run pytest -m integration -q
```

## Style

```bash
uv run ruff check .
uv run ruff format .
```

## Pull requests

- Describe the change and how you verified it (tests run, manual CLI check).
- Keep scope focused; avoid unrelated refactors.
