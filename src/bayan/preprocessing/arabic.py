import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


DISPLAY_PROFILE = ArabicProfile(name="display", dediacritize=False)
MODEL_PROFILE = ArabicProfile(name="bayan_ar_v1", dediacritize=True)

_TATWEEL = "\u0640"
_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670]")
_ALEF_VARIANTS_RE = re.compile(r"[إأآ]")
_ALEF_MAKSURA = "ى"
_TA_MARBUTA = "ة"
_WAW_HAMZA = "ؤ"
_YEH_HAMZA = "ئ"


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_TATWEEL, "")
    if profile.dediacritize:
        text = _DIACRITICS_RE.sub("", text)
    text = _ALEF_VARIANTS_RE.sub("ا", text)
    text = text.replace(_ALEF_MAKSURA, "ي")
    text = text.replace(_TA_MARBUTA, "ه")
    text = text.replace(_WAW_HAMZA, "و")
    text = text.replace(_YEH_HAMZA, "ي")
    return text


def segment(text: str) -> list[str]:
    raise NotImplementedError
