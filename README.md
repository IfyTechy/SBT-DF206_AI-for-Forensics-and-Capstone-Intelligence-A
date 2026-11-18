# ICDFA CIP-B107 LAB01: AI-Assisted IP Evidence Identification and Model Evaluation

**Case ID:** ICDFA-DFIR-AI-026

**Course:** AI for Forensics and Capstone Intelligence A (ICDFA CIP-B107)

**Student:** Nebeuwa Ifeanyichukwu Raphael

**Student / registration ID:** `202520850LE`

**Mode:** Individual practical assessment

> The data used here is a synthetic teaching dataset. Nothing in this repository establishes that any device was compromised or that any address is malicious.

## 1. Purpose

Recovered strings from a simulated mobile-device configuration export contain genuine IPv4 addresses, version numbers, malformed numbers and address-like fragments. This project builds and validates an AI-assisted triage workflow that separates candidate IP evidence from misleading text, and keeps three concepts apart:

| Concept | Question it answers |
| --- | --- |
| Regex match | Does the string look like an IP? |
| Strict valid IPv4 (`ipaddress.IPv4Address`) | Is it syntactically a valid IPv4 address? |
| Source-labelled positive | Did the dataset annotator treat it as an intended IP evidence candidate? |

Disagreements between these three are documented and explained, not deleted or silently relabelled.

## 2. Repository structure

```
.
├── README.md
├── requirements.txt              # pinned package versions
├── evidence_manifest.csv         # chain-of-custody: SHA-256, size, WAT/UTC time, examiner
├── candidate_review.csv          # source_row, context, candidate, regex_match, ipv4_valid, source_label, analyst_note
├── predictions.csv               # actual vs predicted for every holdout record
├── notebooks/                    # executable notebooks / scripts for Phases A-E
├── charts/                       # confusion matrix, loss/accuracy plots
├── report/                       # signed forensic report (PDF) and executive briefing
├── screenshots/                  # timestamped screenshots of my own runs
└── originals/                    # NOT committed (see section 4)
```

## 3. Environment setup

Python 3.10+ is recommended. No GPU and no paid API account is needed.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas scikit-learn matplotlib notebook
pip freeze > requirements.txt
```

## 4. Evidence handling

- The supplied files are copied into an `originals/` folder and are never modified: `dataset_small.csv`, `dataset_train.jsonl`, `dataset_test.jsonl`, `result.csv`, `fine_tuning_steps.txt`.
- `originals/` is listed in `.gitignore`. The supplied scripts and notebooks include embedded credentials and are course material, so they are not republished. Integrity is shown by the SHA-256 hashes in `evidence_manifest.csv`.
- To reproduce: place the supplied files in `originals/`, then run the notebooks in order. The hashes must match `evidence_manifest.csv`.
- The historical upload, create-job and prediction scripts were **not executed**. They use obsolete API patterns and contain embedded credentials. No API keys, tokens or secrets are stored in this repository.

## 5. Workflow (by phase)

| Phase | Work | Output |
| --- | --- | --- |
| A. Preserve and profile | Hashing, chain of custody, dataset profiling, JSONL validation (183 train / 46 test) | `evidence_manifest.csv`, profiling notebook |
| B. Candidate extraction | Regex extractor plus independent strict IPv4 validator, with context retained | `candidate_review.csv` |
| C. Offline classifier | TF-IDF + Logistic Regression (fitted on train only), rule-based baseline, fixed seed, leakage check | model notebook, `predictions.csv` |
| D. Historical fine-tuning evidence | Analysis of `fine_tuning_steps.txt` and `result.csv` (historical, not a new run) | loss/accuracy charts |
| E. Evaluation and interpretation | Confusion matrix, accuracy/precision/recall/F1, majority-class baseline, 8-record analyst review | confusion matrix, review table |
| F. Intelligence report | Forensic memorandum and executive briefing | `report/` |

## 6. Results

*To be completed from my own saved outputs. Every figure below must be backed by `predictions.csv` or by a historical file from `originals/`.*

| Item | Value | Source |
| --- | --- | --- |
| Records in `dataset_small.csv` | 229 | Phase A profile |
| Train / test split | 183 / 46 | JSONL validation |
| Offline classifier metrics (own run) | _pending_ | `predictions.csv` |
| Rule-based baseline metrics (own run) | _pending_ | `predictions.csv` |
| Majority-class baseline | _pending_ | `predictions.csv` |
| Historical fine-tuning metrics (supplied, not my run) | _pending_ | `result.csv` |

## 7. Known limitations

- Small dataset with a class imbalance (positives outnumber negatives).
- Duplicate rows, and identical prompts that appear in both train and test (leakage risk).
- The supplied split is sequential, not randomised.
- Source labels are annotator intent and can disagree with strict IPv4 validity.
- Historical fine-tuning results used a legacy model and endpoints and cannot be assumed reproducible today.
- Recommended use is analyst assistance only. No address is classified as malicious without corroborating evidence.

## 8. Academic integrity

All analysis and code in this repository is my own work on the supplied synthetic data. Historical outputs are clearly separated from newly generated results. No credentials are included.
