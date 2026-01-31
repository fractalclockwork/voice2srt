import sys
import math
from transformers import pipeline


def to_srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - math.floor(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(chunks, path="output.srt"):
    with open(path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            start, end = chunk["timestamp"]
            text = chunk["text"].strip()

            f.write(f"{i}\n")
            f.write(f"{to_srt_timestamp(start)} --> {to_srt_timestamp(end)}\n")
            f.write(text + "\n\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run transcribe.py <audiofile>")
        sys.exit(1)

    audio_path = sys.argv[1]

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
    write_srt(chunks, "output.srt")

    print("Done. Wrote output.srt")


if __name__ == "__main__":
    main()
