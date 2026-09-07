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

## Lab 2 — Parameter audit

| Bucket | mBERT | mBERT % | CAMeLBERT | CAMeLBERT % |
|---|---:|---:|---:|---:|
| Total | 177,853,440 | 100% | 109,081,344 | 100% |
| Embeddings | 92,208,384 | 51.85% | 23,436,288 | 21.49% |
| Attention | 28,348,416 | 15.94% | 28,348,416 | 25.99% |
| FFN | 56,669,184 | 31.86% | 56,669,184 | 51.95% |
| Norms | 36,864 | 0.02% | 36,864 | 0.03% |
| Pooler | 590,592 | 0.33% | 590,592 | 0.54% |

Why is the embedding share different? mBERT covers 104 languages so it needs a much larger vocabulary, which inflates its embedding table; CAMeLBERT is Arabic-only with a smaller vocabulary, so the same-sized attention/FFN layers make up a much bigger share of a smaller total.

# Causal mask
- Applied a lower-triangular mask (torch.tril) so position i attends only to positions ≤ i.
- Verified: upper triangle of the attention weight matrix is all zeros (torch.allclose check = True).
- Model family: Decoder-style causal attention (e.g. GPT-style models).

# Attention diagnostics
- Adjacency head: layer 12 (last), head 8 — avg local attention mass 0.877 (attends mostly to itself + immediate neighbours, n-gram-like behaviour).
- [SEP] sink: average attention mass directed at [SEP] across all layers/heads = 0.129 (~13%), even though [SEP] carries no content — known BERT attention-sink pattern.
- Pad leakage: with a correct attention_mask, pad attention mass = 0.0000. Without any mask (all-ones), pad mass = 0.1544 — 15.44% of total attention wasted on [PAD] tokens, mostly hurting the shorter English example.
- Takeaway: always pass attention_mask at inference/training; skipping it silently degrades short sequences the most.

# Lab 4 — Dialect audit
Distribution:
One-sentence implication for MSA-only evaluation:
