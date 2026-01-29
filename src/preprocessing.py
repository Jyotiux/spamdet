"""Text preprocessing utilities for SMS spam classification.

Includes: lowercase, punctuation removal, stopword removal.
"""
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/non-alphanumeric, and remove stopwords.

    Args:
        text: raw message string

    Returns:
        cleaned string suitable for vectorization
    """
    if not isinstance(text, str):
        text = str(text)
    # lowercase
    text = text.lower()
    # remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # keep only words and numbers
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # tokenize (simple split) and remove stopwords
    tokens = [t for t in text.split() if t and t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def batch_clean(texts):
    """Apply clean_text over an iterable of texts."""
    return [clean_text(t) for t in texts]
