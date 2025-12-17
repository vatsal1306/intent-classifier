import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text_for_model(text: str) -> str:
    # Keep meaning; just normalize unicode + whitespace.
    text = unicodedata.normalize("NFKC", str(text))
    return normalize_whitespace(text)
