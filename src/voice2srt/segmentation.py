from tqdm import tqdm

_NLP = None


def _get_nlp():
    """Load spaCy lazily so its English model isn't sitting in RAM during the
    GPU-heavy Whisper transcription phase."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def sentence_segment(words):
    """
    Convert word-level timestamped tokens into sentence-level groups.
    """
    text = " ".join(w["text"] for w in words)
    doc = _get_nlp()(text)

    sentences = []
    idx = 0

    for sent in tqdm(doc.sents, desc="Segmenting sentences", unit="sentence"):
        sent_words = []
        sent_tokens = sent.text.strip().split()

        for _ in sent_tokens:
            sent_words.append(words[idx])
            idx += 1

        sentences.append(sent_words)

    return sentences


def sentence_timestamp(words):
    return words[0]["timestamp"][0], words[-1]["timestamp"][1]


def split_for_readability(sentences, max_chars=80, max_duration=6.0):
    """
    Break long sentences into smaller subtitle chunks based on:
    - max characters
    - max duration
    """
    final_segments = []

    for sent_words in tqdm(sentences, desc="Splitting for readability", unit="segment"):
        text = " ".join(w["text"] for w in sent_words)
        start = sent_words[0]["timestamp"][0]
        end = sent_words[-1]["timestamp"][1]
        duration = end - start

        if len(text) <= max_chars and duration <= max_duration:
            final_segments.append((text, start, end))
            continue

        # Walk the sentence with running counters so we don't rebuild the
        # joined string on every word (was O(N^2) in transient allocations).
        current = []
        current_chars = 0
        for w in sent_words:
            w_text = w["text"]
            extra = len(w_text) + (1 if current else 0)

            if current:
                chunk_start = current[0]["timestamp"][0]
                if (
                    current_chars + extra > max_chars
                    or w["timestamp"][1] - chunk_start > max_duration
                ):
                    chunk_text = " ".join(x["text"] for x in current)
                    final_segments.append(
                        (chunk_text, chunk_start, current[-1]["timestamp"][1])
                    )
                    current = [w]
                    current_chars = len(w_text)
                    continue

            current.append(w)
            current_chars += extra

        if current:
            chunk_text = " ".join(x["text"] for x in current)
            final_segments.append(
                (
                    chunk_text,
                    current[0]["timestamp"][0],
                    current[-1]["timestamp"][1],
                )
            )

    return final_segments
