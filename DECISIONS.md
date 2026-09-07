# Decision Records

## tokenizer
- Chosen checkpoint(s): xlm-roberta-base
- Arabic fertility evidence: XLM-R = 1.67, second-best after CAMeLBERT (1.41) and far better than mBERT (2.15) and DistilBERT (4.53)
- English fertility evidence: XLM-R = 1.43, the best among all four candidates
- p95 length evidence: XLM-R = 21.0 (AR) / 23.0 (EN), lowest combined sequence length among the balanced candidates
- Operational trade-off / rationale: CAMeLBERT is stronger on Arabic alone, but Bayan is bilingual and needs one shared tokenizer for both languages. XLM-R is the only candidate with no clear weak language — it avoids CAMeLBERT's poor English fertility (2.70) and DistilBERT's unusable Arabic fertility (4.53), so it is the safer single-model choice for the shared preprocessing/training pipeline.

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
