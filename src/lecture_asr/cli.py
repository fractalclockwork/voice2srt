import sys
from .pipeline import transcribe
from .segmentation import sentence_segment, split_for_readability
from .srt import write_srt


def main():
    if len(sys.argv) < 2:
        print("Usage: lecture-asr <audiofile>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"Transcribing {audio_path}...")

    result = transcribe(audio_path)
    words = result["chunks"]

    print("Aligning sentences...")
    sentences = sentence_segment(words)

    print("Applying readability rules...")
    segments = split_for_readability(sentences)

    print("Writing output.srt...")
    write_srt(segments, "output.srt")

    print("Done.")
