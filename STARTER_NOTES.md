# Starter reconstruction notes

The shared course materials describe `lab1-start` … `lab7-start`, checkpoint branches and provided skeletons/tests, but the independently usable Git starter repository/branches were not included in the supplied materials. This repository is therefore a **student starter reconstruction** from the Instructor Package plus the supplied Bayan datasets/fixtures.

Principles used here:
- preserve the guide's lab objectives, targets, evidence and commit checkpoints;
- provide TODO skeletons rather than participant solutions;
- include supplied official data fixtures;
- make Lab 1 explicit: 25 golden preprocessing pairs plus a separate 60-case PII masking recall check;
- use one evolving repository so Labs 1–7 assemble into the final Bayan service;
- avoid copying course reference benchmark values into participant evidence tables.

## Colab readiness
- Added `notebooks/00_colab_setup.ipynb` (environment/setup only; no lab solutions).
- Lab 3 classifier and NER training skeletons accept `--output-dir` so the same scripts can write locally or to mounted Google Drive.
- The repository remains the single source of truth; Lab 7 CPU evidence must not be replaced by GPU timings.
