import argparse
import math
from pathlib import Path

from transformers import pipeline


def to_srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - math.floor(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            start, end = chunk["timestamp"]
            text = chunk["text"].strip()

            f.write(f"{i}\n")
            f.write(f"{to_srt_timestamp(start)} --> {to_srt_timestamp(end)}\n")
            f.write(text + "\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Minimal Whisper → SRT (chunk timestamps; no sentence polish).",
    )
    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument("output", type=Path, help="Output .srt file path")
    args = parser.parse_args()

    audio_path = str(args.audio)
    output_path = args.output

    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-medium",
        chunk_length_s=30,
        return_timestamps=True,
        device="cuda:0",
    )

    print(f"Transcribing {audio_path}...")
    result = pipe(audio_path)

    chunks = result["chunks"]
    write_srt(chunks, output_path)

    print(f"Done. Wrote {output_path}")


if __name__ == "__main__":
    main()
