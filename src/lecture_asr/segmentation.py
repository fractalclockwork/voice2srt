import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


def sentence_segment(words):
    """
    Convert word-level timestamped tokens into sentence-level groups.
    """
    text = " ".join(w["text"] for w in words)
    doc = nlp(text)

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

        # break long sentences into smaller timed chunks
        current = []
        for w in sent_words:
            current.append(w)
            chunk_text = " ".join(x["text"] for x in current)
            chunk_start = current[0]["timestamp"][0]
            chunk_end = current[-1]["timestamp"][1]

            if len(chunk_text) > max_chars or (chunk_end - chunk_start) > max_duration:
                prev = current[:-1]
                prev_text = " ".join(x["text"] for x in prev)
                prev_start = prev[0]["timestamp"][0]
                prev_end = prev[-1]["timestamp"][1]
                final_segments.append((prev_text, prev_start, prev_end))
                current = [w]

        if current:
            chunk_text = " ".join(x["text"] for x in current)
            chunk_start = current[0]["timestamp"][0]
            chunk_end = current[-1]["timestamp"][1]
            final_segments.append((chunk_text, chunk_start, chunk_end))

    return final_segments
