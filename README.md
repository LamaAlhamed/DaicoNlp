# Bayan | بيان
## SDA-AIE-211 — Natural Language Processing with Transformers

> **From raw bilingual text to a working NLP service — one lab at a time.**
>
> في هذا المشروع ما راح نبني 7 تمارين منفصلة. راح نطوّر **Bayan** خطوة بخطوة: من raw Arabic/English text، إلى preprocessing وTransformers وfine-tuning وsemantic search وevaluation، ثم نختم بخدمة FastAPI محسّنة وقابلة للقياس.

---

# 🚀 What are we building?

**Bayan (بيان)** is a bilingual citizen-feedback intelligence service for Arabic and English text.

By the end of the course, this repository should be able to:

- 🧹 preprocess Arabic + English text consistently
- 🔐 mask PII before model use
- 🧠 classify feedback topics/sentiment
- 🏷️ extract entities with NER
- ❓ handle extractive QA with honest no-answer behaviour
- 🔎 retrieve similar historical cases using semantic search
- 📊 evaluate models with slices, confidence intervals, and behavioural tests
- ⚡ optimise inference with ONNX + INT8
- 🌐 serve the final pipeline through FastAPI

```text
Raw Citizen Feedback
        ↓
Versioned Preprocessing
        ↓
Topic Classification / NER / QA
        ↓
Arabic-aware Model Decisions
        ↓
Semantic Search → FAISS → Re-ranking
        ↓
Evaluation + Model Cards + Benchmarks
        ↓
ONNX / INT8 Optimisation
        ↓
FastAPI Bayan Service
```

---

# 🗺️ Course Roadmap

| Day | Lab | Main outcome |
|---|---|---|
| **Day 1** | Lab 1 | Bilingual preprocessing + tokenizer decision |
| **Day 1** | Lab 2 | Transformer attention from scratch + diagnostics |
| **Day 2** | Lab 3A | Topic classifier that beats TF-IDF baseline |
| **Day 2** | Lab 3B | NER + extractive QA |
| **Day 3** | Lab 4 | Arabic normalisation + dialect-aware model |
| **Day 3** | Lab 5 | Bilingual semantic search |
| **Day 3** | Lab 6 | Honest evaluation report + model cards |
| **Day 4** | Lab 7 | ONNX / INT8 optimisation + serving |
| **Day 4** | Capstone | Integrate everything into one Bayan service |

> **Important:** Keep yesterday's work. Every lab builds evidence and components that later labs reuse.

---

# 📁 Repository Structure

```text
SDA-AIE-211-Bayan/
│
├── src/bayan/
│   ├── preprocessing/
│   │   ├── core.py             # Lab 1
│   │   ├── segmentation.py     # Lab 1
│   │   └── arabic.py           # Lab 4
│   ├── attention.py            # Lab 2
│   ├── models/
│   │   ├── data.py             # Lab 3A
│   │   ├── ner.py              # Lab 3B
│   │   └── qa.py               # Lab 3B
│   ├── search/
│   │   ├── index.py            # Lab 5
│   │   └── service.py          # Lab 5
│   ├── evaluation/
│   │   ├── bootstrap.py        # Lab 6
│   │   ├── slices.py           # Lab 6
│   │   └── behavioural.py      # Lab 6
│   └── serving/
│       ├── api.py              # Lab 7 + Capstone
│       └── canaries.py         # Lab 7 + Capstone
│
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_tokenizer_audit.py   # Lab 1
│   ├── 02_transformer_anatomy.py # Lab 2
│   └── 05_retrieval_eval.py    # Lab 5
│
├── scripts/
│   ├── doctor.py
│   ├── parameter_audit.py      # Lab 2
│   ├── tfidf_baseline.py       # Lab 3A
│   ├── train_classifier.py     # Lab 3A
│   ├── train_ner.py            # Lab 3B
│   ├── qa_smoke.py             # Lab 3B
│   ├── dialect_audit.py        # Lab 4
│   ├── arabic_bakeoff.py       # Lab 4
│   ├── evaluation_report.py    # Lab 6
│   ├── benchmark_inference.py  # Lab 7
│   ├── export_onnx.py          # Lab 7
│   └── load_test.sh            # Lab 7
│
├── tests/
├── data/
├── artifacts/
├── templates/
│   └── model_card.md.j2
│
├── NOTES.md
├── BENCHMARKS.md
├── DECISIONS.md
├── EVALUATION_REPORT.md
├── requirements.txt
├── pyproject.toml
└── Makefile
```

---

# ⚙️ First-Time Setup — Do This Once

The course targets **Python 3.12**.

## macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts/doctor.py
```

## Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts/doctor.py
```

Expected final environment message:

```text
ALL GOOD
```

## Every time you reopen the project

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

---

# 🧪 How to Work Through Every Lab

This is a **starter repository**, so some files intentionally contain:

```python
# TODO(Lab X)
raise NotImplementedError(...)
```

That means the repository is waiting for **your implementation**.

Use the same pattern in every lab:

```text
1. Read the lab section below
2. Open the exact file(s) listed under EDIT
3. Run the test/script under RUN BEFORE
4. Observe the failure/baseline
5. Implement the TODOs
6. Run again
7. Record YOUR measured evidence
8. Commit + push
```

> Do not edit expected test values just to make tests pass. The tests are part of the contract.

---

# 🥇 LAB 1 — Bilingual Preprocessing and Tokenisation Pipeline

**Duration:** ~50 minutes  
**Goal:** Build Bayan's shared preprocessing module, verify PII masking, segment sentences, compare four tokenizers, then make a tokenizer decision from measured evidence.

## Lab 1 — What you will edit

```text
NOTES.md
src/bayan/preprocessing/core.py
src/bayan/preprocessing/segmentation.py
notebooks/01_tokenizer_audit.py
BENCHMARKS.md
DECISIONS.md
```

## Lab 1 — Step 1: Defect Safari

### OPEN

```text
data/raw/bayan_raw_sample.csv
```

Inspect the sample before writing cleaning rules.

Find and document at least **6 defect classes** in:

```text
NOTES.md
```

Look for:

```text
Unicode forms
Tatweel / ـ
Code-switching Arabic ↔ English
PII
Emoji
HTML remnants
```

### RUN

No model is required here. You are inspecting the supplied sample and writing observations.

---

## Lab 1 — Step 2: Implement preprocessing

### EDIT

```text
src/bayan/preprocessing/core.py
```

Complete:

```python
normalize(text)
mask_pii(text)
preprocess(text)
```

The implementation must follow the course preprocessing contract. Among the behaviours tested are Unicode normalisation, tatweel removal, repeated-character handling, whitespace cleanup, phone masking, national-ID-shaped masking, and preserving useful signal such as emoji.

### RUN BEFORE IMPLEMENTATION

```bash
pytest tests/test_preprocessing.py -q
```

The starter is expected to fail because the functions still contain TODOs.

### RUN AFTER IMPLEMENTATION

```bash
pytest tests/test_preprocessing.py -q
```

### TARGET

```text
25 passed
```

---

## Lab 1 — Step 3: Verify PII recall

The dedicated PII fixture contains **60 cases**.

### RUN

```bash
pytest tests/test_pii_recall.py -q
```

### TARGET

```text
60/60 PII cases handled correctly
PII recall = 100%
```

> The pytest output may show one aggregate test passing; that test internally checks all 60 rows.

---

## Lab 1 — Step 4: Sentence segmentation

### EDIT

```text
src/bayan/preprocessing/segmentation.py
```

Complete:

```python
build_pipeline()
split_sentences(raw, nlp)
```

Build the spaCy segmentation pipeline and spot-check **5 flagged long examples**, including the numbered-list complaint.

Your segmentation should produce sensible sentence boundaries and should not blindly break useful abbreviations.

### VERIFY

Use the functions interactively or from your editor/terminal to inspect the 5 examples. Record noteworthy findings in:

```text
NOTES.md
```

---

## Lab 1 — Step 5: Tokenizer audit

### EDIT

```text
notebooks/01_tokenizer_audit.py
```

Complete:

```python
fertility(tokenizer, texts)
main()
```

The audit compares these four candidates:

```text
bert-base-multilingual-cased
xlm-roberta-base
CAMeL-Lab/bert-base-arabic-camelbert-mix
distilbert-base-uncased
```

Measure:

```text
AR fertility
EN fertility
sequence lengths
p95 sequence length per language/tokenizer
```

### RUN

```bash
python notebooks/01_tokenizer_audit.py
```

The first run may download tokenizer files, so allow extra time.

### RECORD

Put **your measured numbers** in:

```text
BENCHMARKS.md
```

---

## Lab 1 — Step 6: Make the tokenizer decision

### EDIT

```text
DECISIONS.md
```

Complete the tokenizer decision using measured evidence:

```text
Chosen checkpoint(s)
Arabic fertility
English fertility
p95 length
Why this choice fits Bayan
```

Do not choose based on model popularity alone.

---

## Lab 1 — Final check

macOS/Linux with Make:

```bash
make lab1
```

Cross-platform equivalent:

```bash
pytest tests/test_preprocessing.py tests/test_pii_recall.py -q
```

Then run the tokenizer audit one final time:

```bash
python notebooks/01_tokenizer_audit.py
```

## Lab 1 — Commit

```bash
git status
git add .
git commit -m "feat(preprocessing): versioned bilingual pipeline with tokenizer audit"
git push
```

✅ **Lab 1 is complete when:** preprocessing tests pass, PII recall is 100%, segmentation is spot-checked, tokenizer metrics are recorded, and `DECISIONS.md` contains the tokenizer choice.

---

# 🧠 LAB 2 — Anatomy of a Transformer

**Duration:** ~50 minutes  
**Goal:** Implement scaled dot-product attention and Multi-Head Attention, compare against PyTorch, audit parameter counts, build a causal mask, and diagnose attention-mask leakage.

## Lab 2 — What you will edit

```text
src/bayan/attention.py
notebooks/02_transformer_anatomy.py
scripts/parameter_audit.py
NOTES.md
BENCHMARKS.md
```

## Lab 2 — Step 1: Attention from scratch

### EDIT

```text
src/bayan/attention.py
```

Complete:

```python
attention(q, k, v, mask=None)
MultiHeadAttention
```

Your attention flow should conceptually follow:

```text
Q × Kᵀ
  ↓
scale by √d_k
  ↓
apply mask to scores
  ↓
softmax
  ↓
weights × V
```

### RUN BEFORE / DURING IMPLEMENTATION

```bash
pytest tests/test_attention.py -q
```

### TARGET

Your implementation should match the PyTorch reference to:

```text
atol = 1e-6
```

---

## Lab 2 — Step 2: Wire the transformer anatomy script

### EDIT

```text
notebooks/02_transformer_anatomy.py
```

Use your attention implementation to:

```text
verify numerical equivalence
print/inspect the attention weight matrix
exercise Multi-Head Attention
verify masking behaviour
produce evidence needed by the lab
```

### RUN

```bash
python notebooks/02_transformer_anatomy.py
```

The starter will raise `NotImplementedError` until you complete its TODO.

---

## Lab 2 — Step 3: Parameter audit

### EDIT

```text
scripts/parameter_audit.py
```

Complete:

```python
audit(checkpoint)
```

Audit:

```text
bert-base-multilingual-cased
CAMeL-Lab/bert-base-arabic-camelbert-mix
```

Compare parameter buckets such as embeddings, attention, FFN, norms, pooler, and other parameters.

### RUN

```bash
python scripts/parameter_audit.py
```

### RECORD

Write the comparison in:

```text
NOTES.md
```

Also answer in one sentence:

```text
Why is the embedding share different?
```

Hint: vocabulary size / multilingual tax.

---

## Lab 2 — Step 4: Causal mask

### EDIT

Use:

```text
src/bayan/attention.py
notebooks/02_transformer_anatomy.py
```

Modify the mask so token position `i` can attend only to positions `≤ i`.

Verify the attention matrix is **lower triangular**.

Record which model family this corresponds to:

```text
Decoder-style causal attention
```

### RUN

```bash
python notebooks/02_transformer_anatomy.py
```

---

## Lab 2 — Step 5: Attention-map diagnostics + pad leak

In the anatomy script, inspect attention maps on the Bayan examples.

Look for:

```text
an adjacency-looking head
[SEP] sink behaviour
attention paid to [PAD]
```

Compare pad mass with and without a correct attention mask.

Record findings in:

```text
BENCHMARKS.md
NOTES.md
```

---

## Lab 2 — Final check

```bash
pytest tests/test_attention.py -q
python scripts/parameter_audit.py
python notebooks/02_transformer_anatomy.py
```

Or:

```bash
make lab2
```

## Lab 2 — Commit

```bash
git status
git add .
git commit -m "feat(notebooks): attention implementation + parameter audit + pad-leak diagnosis"
git push
```

✅ **Lab 2 is complete when:** attention equivalence passes, MHA works, parameter audit is recorded, causal masking is verified, and the pad-leak diagnosis is documented.

---

# 🤖 LAB 3A — Topic Classification

**Duration:** ~50 minutes  
**Environment:** GPU pool or Google Colab fallback  
**Goal:** Establish a TF-IDF baseline, build a leakage-safe grouped split, fine-tune a topic classifier, and beat the baseline by at least **+8 macro-F1 points**.

## Before Lab 3 — Colab setup if GPU is needed

Open:

```text
notebooks/00_colab_setup.ipynb
```

Use the **same GitHub repository**. Do not create a second codebase in Colab.

Large training artefacts can be written to Google Drive via `--output-dir`.

---

## Lab 3A — What you will edit

```text
src/bayan/models/data.py
scripts/tfidf_baseline.py
scripts/train_classifier.py
BENCHMARKS.md
```

## Lab 3A — Step 1: TF-IDF + LinearSVC baseline

### EDIT

```text
scripts/tfidf_baseline.py
```

Complete the supplied baseline driver.

### RUN

```bash
python scripts/tfidf_baseline.py
```

### RECORD

Write baseline macro-F1 in:

```text
BENCHMARKS.md
```

Reference output is around `0.71`; your report should contain **your run**, not copied reference numbers.

---

## Lab 3A — Step 2: Build grouped splits

### EDIT

```text
src/bayan/models/data.py
```

Complete:

```python
build_topic_dataset(...)
```

Build train/validation/test splits using the supplied grouping field so the same citizen does not leak across splits.

### RUN CONTRACT TEST

```bash
pytest tests/test_model_data.py -q
```

### TARGET

```text
0 citizen overlap across splits
```

The grouped split is a critical graded engineering requirement.

---

## Lab 3A — Step 3: Fine-tune topic classifier

### EDIT

```text
scripts/train_classifier.py
```

Complete the training TODO so it:

```text
loads the grouped dataset
uses the Lab-1 tokenizer/checkpoint decision
fine-tunes the classifier
evaluates it
saves a re-runnable artefact
```

### RUN LOCALLY / GPU MACHINE

```bash
python scripts/train_classifier.py
```

Default output:

```text
artifacts/topic_classifier
```

### RUN WITH A CUSTOM / GOOGLE DRIVE OUTPUT PATH

```bash
python scripts/train_classifier.py --output-dir /content/drive/MyDrive/SDA-AIE-211/artifacts/topic_classifier
```

### RECORD

Put frozen-test metrics and improvement over baseline in:

```text
BENCHMARKS.md
```

### TARGET

```text
Transformer macro-F1 ≥ baseline + 0.08
```

---

## Lab 3A — Final check

```bash
pytest tests/test_model_data.py -q
python scripts/tfidf_baseline.py
python scripts/train_classifier.py
```

✅ **Lab 3A is complete when:** grouped split is leakage-free, baseline is recorded, trained classifier artefact exists, and the frozen-test delta is recorded.

---

# 🏷️ LAB 3B — NER + Extractive QA

**Duration:** ~50 minutes  
**Goal:** Align BIO labels to subwords correctly, fine-tune NER, and implement extractive QA post-processing including honest no-answer handling.

## Lab 3B — What you will edit

```text
src/bayan/models/ner.py
src/bayan/models/qa.py
scripts/train_ner.py
scripts/qa_smoke.py
BENCHMARKS.md
```

## Lab 3B — Step 1: NER label alignment

### EDIT

```text
src/bayan/models/ner.py
```

Complete:

```python
align_labels(word_ids, word_labels)
```

The alignment must correctly map word-level BIO labels to tokenizer subwords and mask non-first pieces as required by the course contract.

### RUN

```bash
pytest tests/test_ner_alignment.py -q
```

### TARGET

```text
8/8 alignment cases pass
```

This includes an Arabic clitic-related case that breaks naive implementations.

---

## Lab 3B — Step 2: Fine-tune NER

### EDIT

```text
scripts/train_ner.py
```

Complete the TODO so it:

```text
reads the supplied CoNLL data
uses align_labels()
fine-tunes AutoModelForTokenClassification
evaluates with seqeval at entity level
saves the NER artefact
```

### RUN

```bash
python scripts/train_ner.py
```

Default output:

```text
artifacts/ner
```

Or with Drive:

```bash
python scripts/train_ner.py --output-dir /content/drive/MyDrive/SDA-AIE-211/artifacts/ner
```

### TARGET

```text
NER entity-level F1 ≥ 0.80
```

Record the result in `BENCHMARKS.md`.

---

## Lab 3B — Step 3: Extractive QA span selection

### EDIT

```text
src/bayan/models/qa.py
```

Complete:

```python
best_span(...)
```

The function must reject invalid spans and support an honest null/no-answer path.

### RUN CONTRACT TEST

```bash
pytest tests/test_qa.py -q
```

---

## Lab 3B — Step 4: QA smoke set

### EDIT

```text
scripts/qa_smoke.py
```

Complete the driver to run the supplied **12-question smoke set**.

### RUN

```bash
python scripts/qa_smoke.py
```

### TARGET

```text
9/9 answerable questions → correct span
3/3 unanswerable questions → answer=None
```

---

## Lab 3B — Final check

```bash
pytest tests/test_ner_alignment.py tests/test_qa.py -q
python scripts/train_ner.py
python scripts/qa_smoke.py
```

Or contract tests together:

```bash
make lab3
```

## Lab 3B — Commit

```bash
git status
git add .
git commit -m "feat(models): NER + extractive QA with honest null handling"
git push
```

✅ **Lab 3 is complete when:** classifier beats baseline, NER alignment is correct, NER meets the target, and QA handles both answerable and unanswerable cases.

---

# 🇸🇦 LAB 4 — Arabic Pipeline and Dialect-Aware Fine-tuning

**Duration:** ~50 minutes  
**Environment:** Lab 3 environment + CAMeL Tools  
**Goal:** Add model-specific Arabic normalisation, audit dialect distribution, integrate clitic segmentation into NER, and compare Arabic-centric checkpoints on the Gulf slice.

## Lab 4 — What you will edit

```text
src/bayan/preprocessing/arabic.py
scripts/dialect_audit.py
scripts/arabic_bakeoff.py
NOTES.md
BENCHMARKS.md
DECISIONS.md
```

## Lab 4 — Setup

If CAMeL data is not installed:

```bash
camel_data -i defaults
```

---

## Lab 4 — Step 1: Arabic normalisation profiles

### EDIT

```text
src/bayan/preprocessing/arabic.py
```

Complete:

```python
normalize_arabic(text, profile)
```

Implement the two course profiles and preserve display text separately from model-normalised text when appropriate.

### RUN

```bash
pytest tests/test_arabic_normalize.py -q
```

### TARGET

```text
30 passed
```

---

## Lab 4 — Step 2: Dialect audit

### EDIT

```text
scripts/dialect_audit.py
```

Complete the dialect audit over the Arabic slice.

### RUN

```bash
python scripts/dialect_audit.py
```

### RECORD

In:

```text
NOTES.md
```

record:

```text
region/dialect distribution
one-sentence implication of evaluating only on MSA
```

---

## Lab 4 — Step 3: Clitic segmentation for NER

### EDIT

```text
src/bayan/preprocessing/arabic.py
```

Complete:

```python
segment(text)
```

Then wire the segmentation choice consistently into the NER data/training path.

Re-evaluate the Day-2 NER model with the segmentation path and record the **LOCATION recall delta**.

### RECORD

```text
BENCHMARKS.md
```

Target improvement is at least about **+4 recall points** for LOCATION.

---

## Lab 4 — Step 4: Arabic model bake-off

### EDIT

```text
scripts/arabic_bakeoff.py
```

Compare:

```text
CAMeLBERT-mix
CAMeLBERT-DA
MARBERT (optional if time allows)
```

Evaluate at least:

```text
all
Gulf slice
MSA slice
```

### RUN

```bash
python scripts/arabic_bakeoff.py
```

### RECORD

```text
BENCHMARKS.md
DECISIONS.md#arabic-model
```

Choose the winner using **slice evidence**, not only aggregate F1.

### TARGET

```text
dialect-aware model ≥ +4 macro-F1 on Gulf slice vs Day-2 model
```

---

## Lab 4 — Final check

```bash
pytest tests/test_arabic_normalize.py -q
python scripts/dialect_audit.py
python scripts/arabic_bakeoff.py
```

Or:

```bash
make lab4
```

## Lab 4 — Commit

```bash
git status
git add .
git commit -m "feat(arabic): normalisation profiles + dialect audit + DA model beats mix on Gulf slice"
git push
```

✅ **Lab 4 is complete when:** 30 golden pairs pass, dialect mix is documented, segmentation impact is measured, and `DECISIONS.md` contains an Arabic-model choice backed by Gulf-slice evidence.

---

# 🔎 LAB 5 — Bilingual Semantic Search

**Duration:** ~50 minutes  
**Goal:** Build a versioned FAISS index over 20k historical cases, retrieve with a bi-encoder, re-rank with a cross-encoder, evaluate retrieval metrics, and tune honest no-result behaviour.

## Lab 5 — What you will edit

```text
src/bayan/search/index.py
src/bayan/search/service.py
notebooks/05_retrieval_eval.py
BENCHMARKS.md
```

## Lab 5 — Data

```text
data/search/bayan_cases.csv
data/search/bayan_queries.jsonl
data/search/bm25_baseline_results.jsonl
```

---

## Lab 5 — Step 1: Build the FAISS index

### EDIT

```text
src/bayan/search/index.py
```

Complete:

```python
build_index(...)
```

It must:

```text
encode the case corpus
L2-normalise vectors
build the FAISS index
persist index + metadata
persist a manifest
pin model/preprocessing versions in the manifest
```

### RUN CONTRACT TEST WHILE IMPLEMENTING

```bash
pytest tests/test_search_contract.py -q
```

The contract validates the expected search/index interface and manifest discipline.

---

## Lab 5 — Step 2: Two-stage search service

### EDIT

```text
src/bayan/search/service.py
```

Complete:

```python
CaseSearch.__init__(...)
CaseSearch.search(...)
```

The search path should:

```text
load/check the manifest
normalise the query consistently
run bi-encoder retrieval
retrieve candidates
cross-encoder re-rank candidates
apply min_score for honest empty results
```

---

## Lab 5 — Step 3: Retrieval evaluation

### EDIT

```text
notebooks/05_retrieval_eval.py
```

Complete the evaluation driver.

It must report:

```text
recall@10 without reranking
MRR@10 without reranking
recall@10 with reranking
MRR@10 with reranking
cross-lingual slice gap
no-answer threshold behaviour
```

### RUN

```bash
python notebooks/05_retrieval_eval.py
```

### TARGETS

```text
recall@10 ≥ 0.80
MRR@10 ≥ 0.70
no-answer correctness ≥ 17/20
```

Record your measured values and stage latency in:

```text
BENCHMARKS.md
```

---

## Lab 5 — Step 4: Planted unnormalised-vector bug

Verify why an index/query path without correct L2 normalisation can return plausible-looking results but collapse retrieval metrics.

The learning objective is:

```text
Do not approve retrieval by eyeballing results.
Use the labelled query set and metrics.
```

Document the diagnosis.

---

## Lab 5 — Final check

```bash
pytest tests/test_search_contract.py -q
python notebooks/05_retrieval_eval.py
```

Or:

```bash
make lab5
```

## Lab 5 — Commit

Use your measured retrieval result in a meaningful commit message, for example:

```bash
git status
git add .
git commit -m "feat(search): two-stage bilingual case search with evaluated reranking"
git push
```

✅ **Lab 5 is complete when:** the index manifest is valid, two-stage search works, retrieval is evaluated, cross-lingual behaviour is measured, and no-answer threshold evidence is recorded.

---

# 📊 LAB 6 — The Evaluation Report

**Duration:** ~50 minutes  
**Goal:** Build the honest report card for Bayan models using confidence intervals, sliced metrics, behavioural tests, hand-read error taxonomy, and model cards.

## Lab 6 — What you will edit

```text
src/bayan/evaluation/bootstrap.py
src/bayan/evaluation/slices.py
src/bayan/evaluation/behavioural.py
scripts/evaluation_report.py
docs/ERROR_TAXONOMY.md
EVALUATION_REPORT.md
BENCHMARKS.md
templates/model_card.md.j2
```

## Lab 6 — Step 1: Bootstrap confidence intervals

### EDIT

```text
src/bayan/evaluation/bootstrap.py
```

Complete:

```python
bootstrap_ci(...)
paired_bootstrap_diff(...)
```

### RUN

```bash
pytest tests/test_evaluation.py -q
```

The tests validate the core evaluation utilities.

Use the paired bootstrap to answer whether a small metric difference is likely signal or noise.

---

## Lab 6 — Step 2: Sliced report

### EDIT

```text
src/bayan/evaluation/slices.py
```

Complete:

```python
sliced_report(...)
```

Include useful slices such as:

```text
language
dialect
class
length
```

Flag small slices rather than pretending their estimates are precise.

### RECORD

Write the two-sentence headline a manager should read in:

```text
EVALUATION_REPORT.md
```

---

## Lab 6 — Step 3: Behavioural tests

### EDIT

```text
src/bayan/evaluation/behavioural.py
```

Complete:

```python
run_behavioural_suite(...)
```

Cover the supplied behavioural test skeletons, including invariance, directional behaviour, and minimum-functionality tests.

### RECORD

```text
BENCHMARKS.md
```

Targets from the course benchmark include approximately:

```text
invariance ≥ 95%
MFT ≥ 90%
```

---

## Lab 6 — Step 4: Hand-read 120 errors

Use:

```text
data/eval/validation_predictions.csv
docs/ERROR_TAXONOMY.md
```

This step is intentionally **manual**.

Read **120 sampled validation errors** in pairs and tag each error.

Extend the taxonomy if an error does not fit existing categories.

Produce:

```text
error-category histogram
top 3 prioritised fixes
predicted metric delta for each fix
```

> Do not replace this step with an automatic script. The human reading is part of the lab.

---

## Lab 6 — Step 5: Generate evaluation report + model cards

### EDIT

```text
scripts/evaluation_report.py
EVALUATION_REPORT.md
templates/model_card.md.j2
```

Complete the report generator so it combines:

```text
bootstrap results
slice results
behavioural rates
error taxonomy summary
top fixes
model-card evidence
```

### RUN

```bash
python scripts/evaluation_report.py
```

### TARGET

```text
3 model cards
known limitations written by hand
sliced evaluation included
behavioural results included
error taxonomy included
```

---

## Lab 6 — Final check

```bash
pytest tests/test_evaluation.py -q
python scripts/evaluation_report.py
```

Or:

```bash
make lab6
```

## Lab 6 — Commit

```bash
git status
git add .
git commit -m "docs(eval): sliced report + taxonomy + model cards"
git push
```

✅ **Lab 6 is complete when:** evaluation utility tests pass, slices and CIs are reported, behavioural rates are measured, 120 errors are manually tagged, and 3 model cards are committed.

---

# ⚡ LAB 7 — Hit the Latency Budget

**Duration:** ~50 minutes  
**Important:** latency evidence is **CPU-based**. Do not use Colab GPU timings as the production latency evidence.  
**Goal:** Measure first, optimise second; then export to ONNX, quantise where justified, measure accuracy tax, wire the winning classifier into FastAPI, and load-test the HTTP path.

## Lab 7 — What you will edit

```text
scripts/benchmark_inference.py
scripts/export_onnx.py
src/bayan/serving/api.py
src/bayan/serving/canaries.py
scripts/load_test.sh
BENCHMARKS.md
```

## Lab 7 — Setup

Pin the CPU thread count used by the lab:

macOS/Linux:

```bash
export OMP_NUM_THREADS=4
```

Ensure serving/ONNX dependencies are installed through `requirements.txt`.

---

## Lab 7 — Step 1: Baseline benchmark FIRST

### EDIT

```text
scripts/benchmark_inference.py
```

Complete:

```python
benchmark(...)
```

The benchmark should include:

```text
warm-up
production length mix from data/serving/bench_mix.npy
p50
p99
pinned thread count
```

Produce at least:

```text
fp32 baseline @ max_length=512 padded
free-wins row using dynamic padding / max_length≈128
```

### RUN

```bash
python scripts/benchmark_inference.py
```

Record every optimisation rung in:

```text
BENCHMARKS.md
```

> Baseline must exist before ONNX/INT8. Otherwise you do not have a denominator for speed-up.

---

## Lab 7 — Step 2: Export classifier to ONNX

### EDIT

```text
scripts/export_onnx.py
```

Implement classifier ONNX export and preserve a rollback fp32 artefact.

### RUN

```bash
python scripts/export_onnx.py
```

Then benchmark again:

```bash
python scripts/benchmark_inference.py
```

Run a paired quality check against fp32 and record the result.

---

## Lab 7 — Step 3: Quantise classifier to INT8

Continue in:

```text
scripts/export_onnx.py
```

Apply the qconfig appropriate for the lab CPU.

Then:

```bash
python scripts/export_onnx.py
python scripts/benchmark_inference.py
```

Record:

```text
p50
p99
speed-up
macro-F1 / quality metric
accuracy tax
confidence interval for tax where applicable
```

### CLASSIFIER TARGETS

```text
bare p99 ≤ 25 ms
speed-up ≥ 6×
classifier quality tax ≤ 1 macro-F1 point
```

---

## Lab 7 — Step 4: Repeat for NER

Use the same export/benchmark path for the NER model.

Compare:

```text
fp32
ONNX fp32
INT8
```

Then make an evidence-based decision about whether NER should actually use INT8.

Do not assume every model should be quantised just because the classifier benefits.

---

## Lab 7 — Step 5: Wire the winner into FastAPI

### EDIT

```text
src/bayan/serving/api.py
```

Implement Lab 7's classification endpoint:

```python
POST /v1/classify
```

The serving path should use the shared preprocessing contract and the selected optimised classifier artefact.

### EDIT CANARIES

```text
src/bayan/serving/canaries.py
```

Complete startup canaries to catch train/serve skew and incompatible artefacts before serving traffic.

### RUN CONTRACT TEST

```bash
pytest tests/test_serving_contract.py -q
```

### START API

```bash
make serve
```

Cross-platform alternative:

```bash
uvicorn bayan.serving.api:app --host 0.0.0.0 --port 8000
```

Check health:

```text
GET http://localhost:8000/health
```

---

## Lab 7 — Step 6: HTTP load test

The course uses `hey` with:

```text
16 concurrent clients
60 seconds
```

Review/update:

```text
scripts/load_test.sh
data/serving/load_test_config.yaml
```

### RUN

```bash
bash scripts/load_test.sh
```

### HTTP TARGET

```text
HTTP p99 ≤ 40 ms at 16 concurrent clients
0 request errors
startup canaries green
```

---

## Lab 7 — Final check

```bash
pytest tests/test_serving_contract.py -q
python scripts/benchmark_inference.py
python scripts/export_onnx.py
```

Start service:

```bash
make serve
```

Then in another terminal:

```bash
bash scripts/load_test.sh
```

Or run the contract test via:

```bash
make lab7
```

## Lab 7 — Commit

Use **your measured numbers** in the commit message if appropriate. Do not copy reference latency.

Example pattern:

```bash
git status
git add .
git commit -m "perf(serving): onnx int8 classifier with measured latency and quality tax"
git push
```

✅ **Lab 7 is complete when:** benchmark ladder is filled, ONNX/INT8 decisions are evidence-based, serving contract passes, classifier is wired to API, canaries are green, and HTTP p99 is measured under load.

---

# 🏁 FINAL CAPSTONE — Assemble Bayan

The capstone is **not a brand-new project**.

It is the integration of what you built in Labs 1–7 plus at least one extension.

## Components you already built

```text
Lab 1 → preprocessing + tokenizer decision
Lab 2 → transformer understanding / diagnostics
Lab 3 → classifier + NER + QA
Lab 4 → Arabic-specific normalisation / dialect evidence
Lab 5 → semantic search
Lab 6 → evaluation + model cards
Lab 7 → optimised serving + canaries
```

## Capstone files to finish

```text
src/bayan/serving/api.py
src/bayan/serving/canaries.py
EVALUATION_REPORT.md
BENCHMARKS.md
DECISIONS.md
docs/CAPSTONE_CHECKLIST.md
```

Finish/integrate endpoints such as:

```text
POST /v1/classify
POST /v1/entities
POST /v1/search
POST /v1/analyse
```

## Full test suite — ONLY at the end

```bash
pytest -q
```

> During early labs, do **not** use full `pytest -q` as your main lab command because future-lab starter tests may still be intentionally unfinished. Use the lab-specific commands above.

## Start the complete service

```bash
make serve
```

Or:

```bash
uvicorn bayan.serving.api:app --host 0.0.0.0 --port 8000
```

## Final evidence

Your repository should tell the engineering story through:

```text
NOTES.md
BENCHMARKS.md
DECISIONS.md
EVALUATION_REPORT.md
model cards
meaningful Git commit history
```

---

# ✅ Quick Command Cheat Sheet

| Lab | Main checks / runs |
|---|---|
| **Lab 1** | `pytest tests/test_preprocessing.py -q` · `pytest tests/test_pii_recall.py -q` · `python notebooks/01_tokenizer_audit.py` |
| **Lab 2** | `pytest tests/test_attention.py -q` · `python scripts/parameter_audit.py` · `python notebooks/02_transformer_anatomy.py` |
| **Lab 3A** | `pytest tests/test_model_data.py -q` · `python scripts/tfidf_baseline.py` · `python scripts/train_classifier.py` |
| **Lab 3B** | `pytest tests/test_ner_alignment.py tests/test_qa.py -q` · `python scripts/train_ner.py` · `python scripts/qa_smoke.py` |
| **Lab 4** | `pytest tests/test_arabic_normalize.py -q` · `python scripts/dialect_audit.py` · `python scripts/arabic_bakeoff.py` |
| **Lab 5** | `pytest tests/test_search_contract.py -q` · `python notebooks/05_retrieval_eval.py` |
| **Lab 6** | `pytest tests/test_evaluation.py -q` · `python scripts/evaluation_report.py` |
| **Lab 7** | `pytest tests/test_serving_contract.py -q` · `python scripts/benchmark_inference.py` · `python scripts/export_onnx.py` · `make serve` · `bash scripts/load_test.sh` |
| **Final** | `pytest -q` · `make serve` |

---

# 🧭 What goes in each evidence file?

## `NOTES.md`
Observations and engineering findings:

```text
defect safari
dialect distribution
parameter audit explanation
debugging findings
error observations
```

## `BENCHMARKS.md`
Numbers from **your actual runs**:

```text
tokenizer fertility / p95
classifier baseline + transformer metrics
NER metrics
Arabic slice metrics
retrieval recall@10 / MRR@10
behavioural rates
latency p50 / p99
accuracy tax
```

## `DECISIONS.md`
Decisions backed by evidence:

```text
which tokenizer/model and why
which Arabic model and why
which serving artefact and why
what trade-off was accepted
```

## `EVALUATION_REPORT.md`
The final honest quality story:

```text
aggregate metrics
slice metrics + CIs
behavioural tests
error taxonomy
limitations
prioritised fixes
```

---

# 🔁 Git Checkpoint Pattern

At the end of each lab:

```bash
git status
git add .
git commit -m "<meaningful lab checkpoint>"
git push
```

Your Git history is part of the engineering evidence. Commit working milestones instead of one giant final upload.

---

# 🌟 The Finish Line

You are not finishing this course with seven notebooks.

You are finishing with a **bilingual NLP engineering project** that demonstrates:

```text
Data → Preprocessing → Transformer Models → Arabic Handling
→ Retrieval → Evaluation → Optimisation → Serving
```

Build it one checkpoint at a time. Measure everything that matters. Keep the evidence. 🚀
