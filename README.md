# voice2srt

A lightweight command‑line tool for generating clean, sentence‑aligned **SRT subtitles** from long‑form audio (lectures, talks, interviews).  
Built on Whisper Medium (local GPU), spaCy sentence segmentation, and a readability‑aware subtitle splitter.

**Repository:** [github.com/fractalclockwork/voice2srt](https://github.com/fractalclockwork/voice2srt)

## Features

- Local Whisper Medium transcription (GPU‑accelerated)
- Word‑level timestamps
- Sentence segmentation via spaCy
- Readability‑aware subtitle splitting (max chars, max duration)
- Clean `.srt` output to a path you choose
- Progress bars for long operations
- Simple CLI: `voice2srt <audiofile> <output.srt>` plus layout and ASR tuning flags

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/fractalclockwork/voice2srt.git
cd voice2srt
```

### 2. Create the environment with uv

```bash
uv venv --python 3.11
uv sync
```

This installs:

- Whisper + Transformers  
- spaCy + `en_core_web_sm`  
- PyTorch (GPU if available)  
- tqdm  
- All project code and CLI entry points  

### 3. Confirm installation

```bash
uv run voice2srt --help
```

You should see the two positional arguments plus optional **subtitle layout** and **speech recognition** groups.

---

## Usage

### Basic subtitling

```bash
uv run voice2srt path/to/audio.mp3 path/to/subtitles.srt
```

This writes the subtitles to **`path/to/subtitles.srt`** (any directory you have permission to write).

### Subtitle layout

Control how long each line is and how long it stays on screen (passed to the readability splitter):

| Flag | Default | Meaning |
|------|---------|--------|
| `--max-chars N` | `80` | Split a cue when the text exceeds this many characters. |
| `--max-duration SEC` | `6` | Split a cue when it would stay visible longer than this many seconds. |

Example — shorter lines and tighter timing:

```bash
uv run voice2srt --max-chars 42 --max-duration 4.0 talk.mp3 subs.srt
```

### Speech recognition

Optional Whisper overrides:

| Flag | Default | Meaning |
|------|---------|--------|
| `--model ID` | `openai/whisper-medium` | Hugging Face model id. |
| `--device DEV` | `cuda:0` | Torch device (`cpu`, `cuda`, `cuda:0`, …). |
| `--language CODE` | *(auto)* | Force recognition language (e.g. `en`). Omit to let Whisper detect. |

Example — CPU and English-only:

```bash
uv run voice2srt --device cpu --language en talk.mp3 subs.srt
```

### Example

```bash
uv run voice2srt audio/lecture.mp3 lecture.srt
```

Output:

```
Transcribing audio/lecture.mp3...
Running Whisper...
Processing Whisper chunks...
Aligning sentences...
Splitting for readability...
Writing lecture.srt...
Done.
```

### Quick test clip

Processing the full reference lecture (~35 minutes) is slow. To exercise the CLI on about one minute of audio, generate a short excerpt from `data/test_audio.mp3` with ffmpeg:

```bash
ffmpeg -y -i data/test_audio.mp3 -t 60 -c copy data/test_audio_1min.mp3
```

If stream copy fails or timestamps look wrong, re-encode to match the source (48 kHz stereo, 128 kb/s):

```bash
ffmpeg -y -i data/test_audio.mp3 -t 60 -ar 48000 -ac 2 -c:a libmp3lame -b:a 128k data/test_audio_1min.mp3
```

Then run:

```bash
uv run voice2srt data/test_audio_1min.mp3 data/my_subtitles.srt
```

Like `data/test_audio.mp3`, the clip is a local artifact (not checked in); recreate it after cloning if you use it for smoke tests.

---

## GPU Verification

To confirm Whisper is using your GPU:

```bash
watch -n 0.5 nvidia-smi
```

You should see a Python process consuming GPU memory and compute.

---

## Project Structure

```
src/
  voice2srt/
    cli.py
    pipeline.py
    segmentation.py
    srt.py
    transcribe.py
    __init__.py
pyproject.toml
README.md
LICENSE
CONTRIBUTING.md
.github/workflows/ci.yml
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Development

### Run the CLI directly

```bash
uv run python -m voice2srt.cli audio.mp3 subtitles.srt
uv run python -m voice2srt.cli --max-chars 60 --device cpu audio.mp3 out.srt
```

### Format / lint

```bash
uv run ruff check .
uv run ruff format .
```

### Packaging

Build wheels/sdists with uv (requires [`LICENSE`](LICENSE) and metadata in [`pyproject.toml`](pyproject.toml)):

```bash
uv build
```

---

