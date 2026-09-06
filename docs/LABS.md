# SDA-AIE-211 — Lab Checklist (Starter Repository)

This repository contains the full 4-day Bayan starter skeleton. Implement TODOs during the course; do not replace expected values/tests with your own outputs.

## Lab 1 — Bilingual preprocessing and tokenisation pipeline
**Build:**
1. Defect safari over the 200-row raw sample; document ≥6 defect classes in `NOTES.md`.
2. Implement `normalize()` and `mask_pii()` and pass the 25 golden preprocessing pairs.
3. Build the spaCy sentence segmentation pipeline and spot-check five flagged long examples.
4. Audit four tokenizers by language: fertility + sequence-length histogram/p95.
5. Write `DECISIONS.md#tokenizer` from measured evidence.

**Evidence:** 25/25 golden tests, PII recall 60/60 = 100%, tokenizer fertility/length table, tokenizer decision.

**Run:**
```bash
pytest tests/test_preprocessing.py -q
pytest tests/test_pii_recall.py -q
python notebooks/01_tokenizer_audit.py
```
**Checkpoint commit:** `feat(preprocessing): versioned bilingual pipeline with tokenizer audit`

## Lab 2 — Anatomy of a transformer
1. Complete scaled dot-product attention and MHA; verify numerical equivalence with PyTorch (`atol=1e-6`).
2. Parameter audit for mBERT and CAMeLBERT; explain embedding-share difference.
3. Add causal mask; verify lower-triangular attention; name the family.
4. Inspect attention maps and diagnose pad-attention leakage.

**Evidence:** numerical asserts green, parameter audit, annotated attention map, pad-mass regression.

**Run:** `pytest tests/test_attention.py -q` plus `python notebooks/02_transformer_anatomy.py`.

**Checkpoint commit:** `feat(notebooks): attention implementation + parameter audit + pad-leak diagnosis`


## GPU / Colab note before Lab 3
Labs 3–4 may require GPU training. Use `notebooks/00_colab_setup.ipynb` to open the **same GitHub repository** in Colab; do not copy/paste a second implementation into the notebook. Save large model artefacts to a mounted Drive path via `--output-dir` when needed. Lab 7 latency evidence is CPU-based.

## Lab 3A — Topic classification
1. Run TF-IDF + LinearSVC baseline and record macro-F1.
2. Build grouped split; prove zero `citizen_group_id` overlap.
3. Fine-tune topic classifier from the Lab-1 tokenizer/model decision.
4. Evaluate once on the frozen test and record the measured delta.

**Target:** ≥ +8 macro-F1 over TF-IDF.

**Checkpoint commit:** use a measured delta in a meaningful `feat(models): ...` message.

## Lab 3B — NER + extractive QA
1. Implement `align_labels`; all 8 alignment tests must pass, including continuation/clitic case.
2. Fine-tune token classification and evaluate with seqeval at entity level.
3. Implement constrained `best_span`; run 12-question smoke set including 3 unanswerable questions.

**Targets:** NER entity-F1 ≥ 0.80; smoke set 9/9 answerable spans + 3/3 nulls.

**Checkpoint commit:** `feat(models): NER + extractive QA with honest null handling`

## Lab 4 — Arabic pipeline and dialect-aware fine-tuning
1. Implement per-model `normalize_arabic`; pass 30 supplied golden pairs.
2. Dialect audit; record distribution + implication for MSA-only evaluation.
3. Wire clitic segmentation consistently and record LOCATION recall delta.
4. Bake-off Arabic-centric models; record all/Gulf/MSA slices and update `DECISIONS.md#arabic-model`.

**Targets:** 30/30 Arabic golden; LOCATION recall ≥ +4 points; dialect-aware model ≥ +4 macro-F1 on Gulf slice.

**Checkpoint commit:** `feat(arabic): normalisation profiles + dialect audit + DA model beats mix on Gulf slice`

## Lab 5 — Bilingual semantic search
1. Build/persist L2-normalised FAISS index + metadata + manifest pinned to model/preprocessing versions.
2. Evaluate stage-1 bi-encoder on labelled queries.
3. Add multilingual cross-encoder reranking; record MRR lift + latency.
4. Slice by query language, measure cross-lingual gap, tune honest empty-result threshold on no-answer queries.
5. Verify planted unnormalised-vector failure using metrics, not eyeballing.

**Targets:** recall@10 ≥ 0.80; MRR@10 ≥ 0.70; no-answer ≥17/20 empty-correct; gap measured.

**Checkpoint commit:** use measured retrieval results in a meaningful `feat(search): ...` message.

## Lab 6 — The evaluation report
1. Implement bootstrap CI + paired bootstrap difference; pass 6 correctness tests.
2. Run sliced report on topic + dialect-aware models.
3. Complete behavioural tests and record pass/failure rates.
4. Hand-read 120 sampled validation errors; extend taxonomy; produce histogram + top-3 fixes with predicted deltas.
5. Generate model cards; write known limitations by hand; complete `EVALUATION_REPORT.md`.

**Targets:** ≥12 slices; invariance ≥95%; MFT ≥90%; 120 lab errors tagged; 3 model cards.

**Checkpoint commit:** `docs(eval): sliced report + taxonomy + model cards`

## Lab 7 — Hit the latency budget
1. Benchmark baseline before optimisation; then dynamic padding/max-length free wins.
2. Export classifier to ONNX and paired-check quality.
3. Quantise classifier INT8; benchmark and measure quality tax with CI.
4. Repeat for NER; make an evidence-based per-model quantisation decision.
5. Wire winning classifier artefact into API; load test HTTP with 16 concurrent clients and startup canaries green.
6. Fill full optimisation ladder in `BENCHMARKS.md`.

**Targets:** classifier bare p99 ≤25 ms; HTTP p99 ≤40 ms (16 concurrent); ≥6× speed-up; classifier quality tax ≤1 macro-F1; rollback fp32 kept.

**Checkpoint commit:** use your measured latency/tax, not the reference numbers.

## Final integration
After Labs 1–7, run all unit/contract tests:
```bash
pytest -q
```
Then start the complete Bayan service:
```bash
make serve
```
The capstone is assembly of lab outputs plus at least one extension, not a separate new project.
