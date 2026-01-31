import threading
import time
import soundfile as sf
from transformers import pipeline
from tqdm import tqdm


def load_asr_pipeline():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        chunk_length_s=30,
        return_timestamps="word",
        device="cuda:0",
    )


def _progress_bar(duration_estimate, stop_event):
    """
    Show a progress bar that fills over the estimated duration.
    Only updates in whole seconds for cleaner UX.
    """
    with tqdm(total=duration_estimate, desc="Running Whisper", unit="sec") as pbar:
        start = time.time()
        last = 0

        while not stop_event.is_set():
            elapsed = int(time.time() - start)  # whole seconds
            delta = elapsed - last

            if delta > 0:
                pbar.update(delta)
                last = elapsed

            time.sleep(0.2)

        # finish the bar cleanly
        remaining = duration_estimate - last
        if remaining > 0:
            pbar.update(remaining)


def transcribe(audio_path):
    # Estimate audio duration
    audio, sr = sf.read(audio_path)
    duration = len(audio) / sr

    # Estimate Whisper processing time (GPU)
    estimate = max(5, duration / 2)  # rough but effective

    stop_event = threading.Event()
    bar_thread = threading.Thread(target=_progress_bar, args=(estimate, stop_event))
    bar_thread.start()

    pipe = load_asr_pipeline()
    result = pipe(audio_path)

    stop_event.set()
    bar_thread.join()

    # Process chunks with a real progress bar
    chunks = result["chunks"]
    processed = []
    for c in tqdm(chunks, desc="Processing Whisper chunks", unit="chunk"):
        processed.append(c)

    result["chunks"] = processed
    return result
