import re
import unicodedata

PREPROC_VERSION = "1.2.0"


def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation."""
    
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove Arabic tatweel
    text = text.replace("ـ", "")

    # Reduce repeated characters/symbols to a maximum of 2
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national IDs."""

    # Saudi phone numbers:
    # 05XXXXXXXX
    # +9665XXXXXXXX
    # 9665XXXXXXXX
    phone_pattern = r"(?<!\d)(?:\+966|966|0)5\d{8}(?!\d)"
    text = re.sub(phone_pattern, "<PHONE>", text)

    # Saudi national ID / Iqama-shaped 10 digit numbers
    # starts with 1 or 2
    national_id_pattern = r"(?<!\d)[12]\d{9}(?!\d)"
    text = re.sub(national_id_pattern, "<NATIONAL_ID>", text)

    return text


def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing."""

    text = mask_pii(text)
    text = normalize(text)

    return text
