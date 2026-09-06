# Final Capstone — Bayan Service Checklist

## Mandatory
- [ ] One versioned bilingual + Arabic-profile preprocessing contract used by train/eval/serve.
- [ ] PII recall 100% on the 60-case fixture.
- [ ] Segmentation consistent train-to-serve; startup skew canaries green.
- [ ] Topic classifier beats TF-IDF baseline by ≥8 macro-F1.
- [ ] Dialect-aware variant has a CI-backed Gulf-slice improvement.
- [ ] NER entity-F1 ≥0.80; clitic/alignment tests green.
- [ ] Extractive QA has honest null handling; required no-answer target met.
- [ ] Versioned FAISS index + manifest + two-stage reranking.
- [ ] recall@10 ≥0.80 and MRR@10 ≥0.70; cross-lingual gap reported.
- [ ] Sliced evaluation: language/dialect/class/length + bootstrap CIs.
- [ ] Behavioural suite ≥95% invariance and ≥90% MFT.
- [ ] ≥100 hand-read errors in final report (Lab 6 works with 120) + top-3 prioritised fixes.
- [ ] One model card per artefact with hand-written limitations.
- [ ] Classifier HTTP p99 ≤40 ms at 16 concurrent on lab CPU.
- [ ] Full optimisation ladder + paired quality taxes; fp32 rollback retained.
- [ ] `DECISIONS.md` explains model-family/checkpoint choices with fertility/slice evidence.
- [ ] Re-runnable scripts; frozen test untouched until final report; participant-owned benchmark numbers.
- [ ] Meaningful four-day commit history.

## Choose at least one extension
- [ ] Dialect router
- [ ] Two-headed topic+sentiment encoder
- [ ] Search-quality upgrade
- [ ] Batch endpoint `/v1/classify:batch`
- [ ] QA over service documents

## Final deliverables
- [ ] Repository with full commit history
- [ ] `make serve` works from a clean clone and startup canaries are green
- [ ] `EVALUATION_REPORT.md`
- [ ] `BENCHMARKS.md`
- [ ] `DECISIONS.md`
- [ ] 5-minute bilingual live demo: classification + entities + similar cases; explain one behavioural test; defend one metric claim
