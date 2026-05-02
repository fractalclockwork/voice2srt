import gc
import threading
import time

import soundfile as sf
import torch
from transformers import pipeline
from tqdm import tqdm

DEFAULT_WHISPER_MODEL = "openai/whisper-medium"
DEFAULT_DEVICE = "cuda:0"


def load_asr_pipeline(*, model: str = DEFAULT_WHISPER_MODEL, device: str = DEFAULT_DEVICE):
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        chunk_length_s=30,
        return_timestamps="word",
        device=device,
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
            elapsed = int(time.time() - start)
            delta = elapsed - last

            if delta > 0:
                pbar.update(delta)
                last = elapsed

            time.sleep(0.2)

        remaining = duration_estimate - last
        if remaining > 0:
            pbar.update(remaining)


def transcribe(
    audio_path: str,
    *,
    model: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_DEVICE,
    language: str | None = None,
):
    info = sf.info(audio_path)
    duration = info.frames / info.samplerate

    # Estimate Whisper processing time (GPU)
    estimate = max(5, duration / 2)

    stop_event = threading.Event()
    bar_thread = threading.Thread(target=_progress_bar, args=(estimate, stop_event))
    bar_thread.start()

    pipe = load_asr_pipeline(model=model, device=device)

    pipe_kwargs: dict = {}
    if language:
        pipe_kwargs["generate_kwargs"] = {"language": language, "task": "transcribe"}

    result = pipe(audio_path, **pipe_kwargs)

    stop_event.set()
    bar_thread.join()

    chunks = result["chunks"]

    # Release the model + pipeline immediately so VRAM is free for the rest of
    # the run (segmentation, SRT writing). Without this the ~3GB whisper-medium
    # weights stay pinned on GPU until the function frame is GC'd.
    del result, pipe
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return chunks
