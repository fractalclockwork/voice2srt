# lecture-asr

A lightweight command‑line tool for generating clean, sentence‑aligned subtitles from long‑form audio (lectures, talks, interviews).  
Built on Whisper Medium (local GPU), spaCy sentence segmentation, and a readability‑aware subtitle splitter.

## Features

- Local Whisper Medium transcription (GPU‑accelerated)
- Word‑level timestamps
- Sentence segmentation via spaCy
- Readability‑aware subtitle splitting (max chars, max duration)
- Clean `.srt` output
- Progress bars for long operations
- Simple CLI: `lecture-asr <audiofile>`

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourname/lecture-asr.git
cd lecture-asr
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
uv run lecture-asr --help
```

You should see the usage message.

---

## Usage

### Basic transcription

```bash
uv run lecture-asr path/to/audio.mp3
```

This produces:

```
output.srt
```

in the working directory.

### Example

```bash
uv run lecture-asr audio/lecture.mp3
```

Output:

```
Transcribing audio/lecture.mp3...
Running Whisper...
Processing Whisper chunks...
Aligning sentences...
Splitting for readability...
Writing output.srt...
Done.
```

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
  lecture_asr/
    cli.py
    pipeline.py
    segmentation.py
    srt.py
    __init__.py
pyproject.toml
README.md
```

---

## Development

### Run the CLI directly

```bash
uv run python -m lecture_asr.cli audio.mp3
```

### Format / lint (optional)

```bash
ruff check .
ruff format .
```

---


