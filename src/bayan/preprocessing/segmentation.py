import spacy

from bayan.preprocessing.core import preprocess


def build_pipeline():
    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    text = preprocess(raw)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


if __name__ == "__main__":
    examples = [
        "  لووووسمحت ألطريق المؤدي إلى حي الياسمين يحتاج صيانة عاجلة 😡 <br> 0551234567 1023456789   ",
        "  My My Licence licence request has been under review for 15 days   ",
        "I need clarification on the documents required for Maintenance Appointments — very frustrating 😡",
        "There is a pothole near Tahlia Street for 21 days — very frustrating 😡",
        "الخدمة سيئة جداً: 1) وقت الانتظار طويل 2) الموظف غير متعاون 3) لا يوجد رد على الشكوى",
    ]

    nlp = build_pipeline()
    for ex in examples:
        print("RAW:", ex)
        print("SENTENCES:", split_sentences(ex, nlp))
        print("---")
