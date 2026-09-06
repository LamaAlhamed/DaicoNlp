# Data Dictionary

## bayan_feedback.csv
`feedback_id`, `created_at`, `citizen_group_id`, `lang`, `dialect_region`, `topic`, `sentiment`, `channel`, `city`, `text`, `split`, `synthetic`. Group by `citizen_group_id` when creating alternative splits.

## bayan_cases.csv
Resolved historical cases for retrieval: `case_id`, `lang`, `topic`, `case_text`, `resolution`, `status`, `closed_at`, `synthetic`.

## bayan_queries.jsonl
Judged retrieval queries. `relevant_case_ids` is a JSON array; `no_answer=true` must produce no result above the tuned threshold.

## NER schema
BIO tags for `SERVICE`, `LOCATION`, `DATE`, `REFERENCE`, and `ORGANISATION`. Continuation subwords and special tokens must be excluded with `-100` during training.

## Split policy
The supplied split is 70/20/10 and deterministic. The final test split is frozen. Never use it for error analysis or threshold tuning.

## Safety
All content is synthetic. PII-shaped strings exist only in `pii_test_set.csv` and the raw defect sample to test masking.
