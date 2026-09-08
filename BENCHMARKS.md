# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.15 | 1.51 | 27.0 | 25.0 | n/a |
| XLM-R | 1.67 | 1.43 | 21.0 | 23.0 | n/a |
| CAMeLBERT | 1.41 | 2.70 | 20.0 | 38.0 | n/a |
| DistilBERT | 4.53 | 1.30 | 47.0 | 21.0 | n/a |

- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60 / 60 = 100%


## Lab 2 — Attention diagnostics
- Pad attention mass WITH mask: 0.0000
- Pad attention mass WITHOUT mask: 0.1544
- Most adjacency-looking head: layer 12, head 8 (avg local mass 0.877)
- Average [SEP] attention mass (all layers/heads): 0.1290
- Causal mask check: lower-triangular = True

## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 | 0.25s |
| Topic classifier | macro-F1 | 1.0000 | 1.0000 | 482.45s |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:
