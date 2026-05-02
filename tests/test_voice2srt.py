"""
Validation tests for voice2srt.

Reference assets (read-only): ``data/test_audio_1min.mp3`` and ``data/test_output_1min.srt``
(the ~1-minute excerpt described in the README).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REF_AUDIO = DATA_DIR / "test_audio_1min.mp3"
REF_SRT = DATA_DIR / "test_output_1min.srt"

skip_without_ref_assets = pytest.mark.skipif(
    not REF_AUDIO.is_file() or not REF_SRT.is_file(),
    reason="missing data/test_audio_1min.mp3 or data/test_output_1min.srt (optional local artifacts)",
)
skip_without_ref_audio = pytest.mark.skipif(
    not REF_AUDIO.is_file(),
    reason="missing data/test_audio_1min.mp3 (optional local artifact)",
)
skip_without_ref_srt = pytest.mark.skipif(
    not REF_SRT.is_file(),
    reason="missing data/test_output_1min.srt (optional local artifact)",
)

TIMESTAMP_RANGE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


def _parse_srt_timestamp(ts: str) -> float:
    hh, mm, ss_ms = ts.split(":")
    ss, ms = ss_ms.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_srt(path: Path) -> list[dict]:
    """Parse a UTF-8 SRT file into cue dicts (index, start, end, text)."""
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    blocks = re.split(r"\n\s*\n+", content)
    cues: list[dict] = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        idx_line = lines[0].strip()
        time_line = lines[1].strip()
        m = TIMESTAMP_RANGE.search(time_line)
        if not m:
            continue
        index = int(idx_line)
        start = _parse_srt_timestamp(m.group("start"))
        end = _parse_srt_timestamp(m.group("end"))
        body_lines = lines[2:]
        text = "\n".join(body_lines).strip()
        cues.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "text": text,
            }
        )
    return cues


def normalize_subtitle_text(s: str) -> str:
    """Collapse all whitespace so golden vs generated spacing differences compare equal."""
    return " ".join(s.split())


def assert_srts_equivalent(actual_path: Path, ref_path: Path, *, time_tol: float = 1e-3) -> None:
    actual = parse_srt(actual_path)
    ref = parse_srt(ref_path)
    assert len(actual) == len(ref), (
        f"cue count mismatch: actual={len(actual)} ref={len(ref)} "
        f"({actual_path} vs {ref_path})"
    )
    for i, (a, r) in enumerate(zip(actual, ref, strict=True)):
        assert abs(a["start"] - r["start"]) <= time_tol, (
            f"cue {i + 1} start: {a['start']} vs {r['start']}"
        )
        assert abs(a["end"] - r["end"]) <= time_tol, (
            f"cue {i + 1} end: {a['end']} vs {r['end']}"
        )
        na = normalize_subtitle_text(a["text"])
        nr = normalize_subtitle_text(r["text"])
        assert na == nr, f"cue {i + 1} text mismatch:\n  actual: {na!r}\n  ref:    {nr!r}"


# --- Fast tests (no GPU, no large downloads) ---


@skip_without_ref_assets
def test_reference_files_exist():
    assert REF_AUDIO.is_file(), f"missing {REF_AUDIO}"
    assert REF_SRT.is_file(), f"missing {REF_SRT}"


@skip_without_ref_srt
def test_reference_srt_parses_coherent_one_minute_clip():
    cues = parse_srt(REF_SRT)
    assert len(cues) >= 3
    assert cues[0]["index"] == 1
    assert cues[-1]["end"] <= 120.0  # ~1 min excerpt


@skip_without_ref_audio
def test_sf_info_duration_matches_audio_file():
    import soundfile as sf

    info = sf.info(REF_AUDIO)
    duration = info.frames / info.samplerate
    assert 50 < duration < 75  # ~1 min clip (README ffmpeg -t 60)


def test_cli_writes_to_given_output_path(tmp_path, monkeypatch):
    from voice2srt.cli import main

    audio = tmp_path / "in.mp3"
    audio.write_bytes(b"")
    out = tmp_path / "custom_subs.srt"
    monkeypatch.setattr(
        "voice2srt.cli.transcribe",
        lambda _path, **_kw: [{"text": "Hello.", "timestamp": (0.0, 1.0)}],
    )
    monkeypatch.setattr("voice2srt.cli.sentence_segment", lambda words: [words])
    monkeypatch.setattr(
        "voice2srt.cli.split_for_readability",
        lambda _sentences, **_kw: [("Hello.", 0.0, 1.0)],
    )
    monkeypatch.setattr(
        "sys.argv",
        ["voice2srt", str(audio), str(out)],
    )
    main()
    assert "Hello." in out.read_text(encoding="utf-8")


def test_cli_exits_when_output_argument_missing(tmp_path, monkeypatch):
    from voice2srt.cli import main

    audio = tmp_path / "in.mp3"
    monkeypatch.setattr("sys.argv", ["voice2srt", str(audio)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_cli_passes_layout_options_to_split_for_readability(tmp_path, monkeypatch):
    from voice2srt.cli import main

    audio = tmp_path / "in.mp3"
    audio.write_bytes(b"")
    out = tmp_path / "out.srt"
    captured: dict = {}

    def spy_split(sentences, max_chars=80, max_duration=6.0):
        captured["max_chars"] = max_chars
        captured["max_duration"] = max_duration
        return [("Hi.", 0.0, 1.0)]

    monkeypatch.setattr(
        "voice2srt.cli.transcribe",
        lambda _path, **_kw: [{"text": "Hi.", "timestamp": (0.0, 1.0)}],
    )
    monkeypatch.setattr("voice2srt.cli.sentence_segment", lambda words: [words])
    monkeypatch.setattr("voice2srt.cli.split_for_readability", spy_split)
    monkeypatch.setattr(
        "sys.argv",
        [
            "voice2srt",
            "--max-chars",
            "42",
            "--max-duration",
            "3.5",
            str(audio),
            str(out),
        ],
    )
    main()
    assert captured == {"max_chars": 42, "max_duration": 3.5}


def test_cli_passes_asr_options_to_transcribe(tmp_path, monkeypatch):
    from voice2srt.cli import main

    audio = tmp_path / "in.mp3"
    audio.write_bytes(b"")
    out = tmp_path / "out.srt"
    calls: list = []

    def fake_transcribe(path, *, model, device, language):
        calls.append({"path": path, "model": model, "device": device, "language": language})
        return [{"text": "Hi.", "timestamp": (0.0, 1.0)}]

    monkeypatch.setattr("voice2srt.cli.transcribe", fake_transcribe)
    monkeypatch.setattr("voice2srt.cli.sentence_segment", lambda words: [words])
    monkeypatch.setattr(
        "voice2srt.cli.split_for_readability",
        lambda s, **kw: [("Hi.", 0.0, 1.0)],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "voice2srt",
            "--model",
            "openai/whisper-tiny",
            "--device",
            "cpu",
            "--language",
            "en",
            str(audio),
            str(out),
        ],
    )
    main()
    assert len(calls) == 1
    assert calls[0]["path"] == str(audio)
    assert calls[0]["model"] == "openai/whisper-tiny"
    assert calls[0]["device"] == "cpu"
    assert calls[0]["language"] == "en"


def test_write_srt_roundtrip(tmp_path):
    from voice2srt.srt import write_srt

    segments = [
        ("Hello world.", 1.5, 3.25),
        ("Second line.", 4.0, 6.0),
    ]
    out = tmp_path / "out.srt"
    write_srt(segments, out)
    cues = parse_srt(out)
    assert len(cues) == 2
    assert normalize_subtitle_text(cues[0]["text"]) == "Hello world."
    assert pytest.approx(cues[0]["start"], abs=1e-6) == 1.5


def test_split_for_readability_short_sentence():
    from voice2srt.segmentation import split_for_readability

    words = [
        {"text": "Short.", "timestamp": (0.0, 1.0)},
    ]
    out = split_for_readability([words])
    assert len(out) == 1
    assert normalize_subtitle_text(out[0][0]) == "Short."


def test_split_for_readability_splits_on_length():
    from voice2srt.segmentation import split_for_readability

    # "aa bb cc" is 8 characters — max_chars=7 forces a split inside the sentence.
    words = [
        {"text": "aa", "timestamp": (0.0, 0.1)},
        {"text": "bb", "timestamp": (0.1, 0.2)},
        {"text": "cc", "timestamp": (0.2, 0.3)},
    ]
    out = split_for_readability([words], max_chars=7, max_duration=10.0)
    assert len(out) == 2
    assert out[0][0] == "aa bb"
    assert out[1][0] == "cc"


# --- Full pipeline vs golden SRT (GPU, slow) ---


@pytest.mark.integration
@skip_without_ref_assets
@pytest.mark.skipif(
    os.environ.get("VOICE2SRT_E2E", "") != "1",
    reason="set VOICE2SRT_E2E=1 to run full Whisper + segmentation test",
)
@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="pipeline is configured for CUDA (see load_asr_pipeline device)",
)
def test_full_pipeline_output_matches_reference_srt(tmp_path):
    from voice2srt.pipeline import transcribe
    from voice2srt.segmentation import sentence_segment, split_for_readability
    from voice2srt.srt import write_srt

    out_srt = tmp_path / "generated.srt"
    words = transcribe(str(REF_AUDIO))
    assert isinstance(words, list) and len(words) > 0
    sentences = sentence_segment(words)
    segments = split_for_readability(sentences)
    write_srt(segments, out_srt)
    assert_srts_equivalent(out_srt, REF_SRT)
