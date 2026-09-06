"""Lab 1 golden preprocessing contract: 25 input/output pairs.

Do not edit expected values to make the test pass. Fix the implementation.
"""
import pytest
from bayan.preprocessing.core import mask_pii, normalize, preprocess


GOLDEN = [
    (normalize, "مـشكلة", "مشكلة"),
    (normalize, "الخدمــــة", "الخدمة"),
    (normalize, "هلووو", "هلوو"),
    (normalize, "ممتااااز", "ممتااز"),
    (normalize, "تممم", "تمم"),
    (normalize, "GOOD!!!", "GOOD!!"),
    (normalize, "  نص   فيه   مسافات  ", "نص فيه مسافات"),
    (normalize, "hello    world", "hello world"),
    (normalize, "سطر\n\nثاني", "سطر ثاني"),
    (normalize, "a\t\tb", "a b"),
    (normalize, "الخدمة 😡", "الخدمة 😡"),
    (normalize, "Service ✅ ممتاز", "Service ✅ ممتاز"),
    (normalize, "مــــمتازززز", "ممتازز"),
    (normalize, "Noooo", "Noo"),
    (normalize, "عاجلللل", "عاجلل"),
    (mask_pii, "اتصل 0551234567", "اتصل <PHONE>"),
    (mask_pii, "Call +966551234567", "Call <PHONE>"),
    (mask_pii, "Call 966551234567", "Call <PHONE>"),
    (mask_pii, "رقم الهوية 1023456789", "رقم الهوية <NATIONAL_ID>"),
    (mask_pii, "ID 2123456789", "ID <NATIONAL_ID>"),
    (preprocess, "  0551234567   😡  ", "<PHONE> 😡"),
    (preprocess, "رقمي +966551234567 والخدمــــة", "رقمي <PHONE> والخدمة"),
    (preprocess, "هوية 1023456789   ممتااااز", "هوية <NATIONAL_ID> ممتااز"),
    (preprocess, "Call me 966551234567 pleaseeee", "Call me <PHONE> pleasee"),
    (preprocess, "  لا توجد بيانات شخصية هنا  ", "لا توجد بيانات شخصية هنا"),
]


@pytest.mark.parametrize("fn,raw,expected", GOLDEN)
def test_preprocessing_golden(fn, raw, expected):
    assert fn(raw) == expected
