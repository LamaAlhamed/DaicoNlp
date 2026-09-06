"""Lab 4 starter: per-model Arabic normalisation profiles."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    # TODO(Lab 4): implement the two course profiles and preserve a separate display copy.
    raise NotImplementedError


def segment(text: str) -> list[str]:
    # TODO(Lab 4): wire the chosen CAMeL Tools clitic segmentation scheme.
    raise NotImplementedError
