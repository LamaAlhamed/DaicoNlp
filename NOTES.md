# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

# Defect 1
Class:HTML remnants
Example: FB-000001,ar,  لووووسمحت ألطريق المؤدي إلى حي الياسمين يحتاج صيانة عاجلة 😡 <br> 0551234567 1023456789
Why it matters:HTML tags add unnecessary noise to the text and may affect NLP processing.
Decision: remove HTML tags.

# Defect 2
Class:PII
Example:FB-000001,ar,  لووووسمحت ألطريق المؤدي إلى حي الياسمين يحتاج صيانة عاجلة 😡 <br> 0551234567 1023456789
Why it matters:Personal Information
Decision:mask

# Defect 3
Class:Repeated characters
Example:لووووسمحت دفعت الفاتورة لكن الحالة ما زالت غير مسددة
Why it matters:Repeated characters create inconsistent word forms and may affect NLP processing.
Decision:normalize repeated characters.

# Defect 4
Class:Emoji
Example: تطبيق بلدي يتوقف عند تسجيل الدخول 😡
Why it matters: Emojis may carry useful emotional meaning but can affect text processing.
Decision:preserve if sentiment is important.

# Defect 5
Class:Code-switching
Example: تم احتساب رسوم غير صحيحة على الفاتورة رقم BYN-2025-000132
Why it matters: Mixing Arabic and English may affect language-specific NLP processing.
Decision:preserve meaningful English identifiers.

# Defect 6
Class:Unicode forms
Example: ألطريق المؤدي إلى طريق الملك فهد يحتاج صيانة عاجلة
Why it matters: Different character forms can create inconsistent text and affect NLP processing.
Decision:normalize Unicode forms.

# Lab 2 — Parameter audit
Checkpoint	Total params	Embeddings %	Other notes
mBERT			
CAMeLBERT		

# Lab 4 — Dialect audit
Distribution:
One-sentence implication for MSA-only evaluation:
