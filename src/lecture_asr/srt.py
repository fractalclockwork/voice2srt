import math


def to_srt_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - math.floor(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (text, start, end) in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{to_srt_timestamp(start)} --> {to_srt_timestamp(end)}\n")
            f.write(text.strip() + "\n\n")
