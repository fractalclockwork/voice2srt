import argparse
from pathlib import Path

from .pipeline import DEFAULT_DEVICE, DEFAULT_WHISPER_MODEL, transcribe
from .segmentation import sentence_segment, split_for_readability
from .srt import write_srt


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _positive_float(value: str) -> float:
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return x


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate clean, sentence-aligned SRT subtitles from audio "
            "(local Whisper, spaCy, readability splitting)."
        ),
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Input audio file",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output .srt file path",
    )

    layout = parser.add_argument_group("subtitle layout")
    layout.add_argument(
        "--max-chars",
        type=_positive_int,
        default=80,
        metavar="N",
        help="Maximum characters per subtitle chunk before splitting (default: 80)",
    )
    layout.add_argument(
        "--max-duration",
        type=_positive_float,
        default=6.0,
        metavar="SEC",
        help="Maximum on-screen duration per chunk in seconds (default: 6)",
    )

    asr = parser.add_argument_group("speech recognition")
    asr.add_argument(
        "--model",
        default=DEFAULT_WHISPER_MODEL,
        metavar="ID",
        help=f"Whisper model id on the Hub (default: {DEFAULT_WHISPER_MODEL})",
    )
    asr.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        metavar="DEV",
        help=f"Torch device for Whisper, e.g. cuda:0 or cpu (default: {DEFAULT_DEVICE})",
    )
    asr.add_argument(
        "--language",
        default=None,
        metavar="CODE",
        help=(
            "Force Whisper language (e.g. en). Omit for automatic language detection."
        ),
    )

    args = parser.parse_args()

    audio_path = str(args.audio)
    output_path = args.output

    print(f"Transcribing {audio_path}...")

    words = transcribe(
        audio_path,
        model=args.model,
        device=args.device,
        language=args.language,
    )

    print("Aligning sentences...")
    sentences = sentence_segment(words)

    print("Applying readability rules...")
    segments = split_for_readability(
        sentences,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
    )

    print(f"Writing {output_path}...")
    write_srt(segments, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
