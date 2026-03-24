<div align="center">

# CTN-0094 ML Pipeline

**A modular, scalable pipeline for statistical modeling and fairness analysis on opioid use disorder treatment data.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Agj-EXE9WhLjMulA1mIVzW4YWNN7fiqX)
&nbsp;
[![DAB Website](https://img.shields.io/badge/DAB-Website-blue)](https://ctn-0094.github.io/Pipeline/)
&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/CTN-0094/Pipeline/blob/main/LICENSE)
&nbsp;
[![NIH AIM-AHEAD](https://img.shields.io/badge/NIH-AIM--AHEAD-green)](https://ctnlibrary.org/protocol/ctn0094/)

</div>

---

## Overview

This project establishes a modular, reproducible data pipeline for statistical and machine learning modeling on the [CTN-0094](https://ctn-0094.github.io/Pipeline/) database. It supports multiple modeling strategies and evaluation metrics to study the relationship between patient demographics and opioid use disorder treatment outcomes — with a focus on algorithmic bias and fairness.

| | |
|---|---|
| **Clinical Lead** | Prof. Laura Brandt (City College of New York) |
| **Computational Lead** | Prof. Gabriel Odom |
| **Data Scientist** | Ganesh Jainarain |
| **Funding** | NIH AIM-AHEAD `1OT2OD032581-02-267` |

---

## Quick Start

```bash
python3 run_pipelineV2.py --data <path/to/data.csv> --outcome <outcome_name> [options]
```

Run `python3 run_pipelineV2.py --help` for all available arguments.

**Examples**

```bash
# Single outcome, single seed
python3 run_pipelineV2.py --data data.csv --outcome ctn0094_relapse_event -d ./results

# Loop through seeds 5–10 across all outcomes
python3 run_pipelineV2.py --data data.csv -l 5 10 -d ./results

# Loop through seeds with a subset of outcomes
python3 run_pipelineV2.py --data data.csv -l 5 10 -o ctn0094_relapse_event Ab_ling_1998 -d ./results

# Preprocess + PSM only — skip model training
python3 run_pipelineV2.py --data data.csv --outcome ctn0094_relapse_event --data_only -d ./results
```

---

## CLI Arguments

| Argument | Default | Description |
|:---|:---:|:---|
| `--data` | *(required)* | Path to cleaned input dataset (CSV) |
| `-o / --outcome` | all | Outcome(s) to run |
| `-l / --loop` | — | Min and max seed for multi-seed runs |
| `-d / --dir` | `""` | Output directory for logs, predictions, and evaluations |
| `--type` | — | Endpoint type for custom outcomes: `logical`, `integer`, `survival` |
| `--majority` | `1` | Value representing the majority group in the PSM column |
| `--split` | `RaceEth` | Column containing the two groups for PSM |
| `--match` | `age is_female` | Columns to match on during PSM |
| `--group_size` | `500` | Size of each PSM group (half the cohort size) |
| `--heldout_size` | `100` | Size of the held-out evaluation set |
| `--heldout_set_percent_majority` | `58` | Percent majority samples in the held-out set |
| `--data_only` | `False` | Save ML-ready datasets without running model training |
| `-p / --prof` | — | Profiling mode: `simple` or `complex` |

---

## Pipeline Architecture

```
Input CSV
    │
    ▼
┌─────────────────────────────┐
│  Step 1 · Validation &      │  Schema checks, column typing, endpoint validation
│           Preprocessing     │  Binary encoding, TLFB feature engineering
└────────────────┬────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│  Step 2 · Propensity Score  │  Majority/minority group splitting
│           Matching (PSM)    │  Balanced cohort construction + stratified held-out set
└────────────────┬────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│  Step 3 · Feature Selection │  L1 (Lasso) regularization
└────────────────┬────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│  Step 4 · Model Training    │  Auto-selected by endpoint type (see table below)
└────────────────┬────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│  Step 5 · Evaluation        │  Subset test split + held-out set, with demographics
└────────────────┬────────────┘
                 │
                 ▼
    Repeat over seeds & outcomes  →  Results directory
```

### Model Selection

| Endpoint Type | Model | Evaluation Metrics |
|:---|:---|:---|
| `logical` — binary | Logistic Regression (L1) | ROC-AUC, Precision, Recall, Confusion Matrix |
| `integer` — count | Negative Binomial Regression | MSE, RMSE, MAE, Pearson r, McFadden R² |
| `survival` — time-to-event | Cox Proportional Hazard | Concordance Index (C-statistic) |

> All evaluations include the demographic makeup of the training cohort for bias auditing.

---

## Available Outcomes

| Outcome | Endpoint Type |
|:---|:---:|
| `ctn0094_relapse_event` | Logical |
| `Ab_krupitskyA_2011` | Logical |
| `Ab_ling_1998` | Logical |
| `Rs_johnson_1992` | Logical |
| `Rs_krupitsky_2004` | Logical |
| `Rd_kostenB_1993` | Logical |
| `Ab_schottenfeldB_2008` | Integer |
| `Ab_mokri_2016` | Survival |

Custom outcomes can be passed with `-o <name> --type <logical|integer|survival>`.

---

## Project Structure

```
Pipeline/
├── run_pipelineV2.py              # Main entry point & CLI
├── src/
│   ├── constants.py               # EndpointType enum
│   ├── validate.py                # Dataset validation
│   ├── data_preprocessing.py      # Feature engineering
│   ├── preprocess_pipeline.py     # Preprocessing orchestration
│   ├── create_demodf_knn.py       # PSM & held-out set construction
│   ├── model_training.py          # Training + evaluation orchestration
│   ├── train_model.py             # Model classes (Logistic, NegBin, CoxPH, Beta)
│   ├── logging_setup.py           # Logging configuration
│   └── utils.py                   # Shared utilities
└── tests/                         # Unit tests
```

---

## Testing

```bash
pytest tests/
```

Test coverage includes input validation, model training, held-out splitting, data ingestion, and constants.

---

## References

Luo SX, Feaster DJ, Liu Y et al. **Individual-Level Risk Prediction of Return to Use During Opioid Use Disorder Treatment.** *JAMA Psychiatry.* 2024;81(1):45–56. [doi:10.1001/jamapsychiatry.2023.3596](https://jamanetwork.com/journals/jamapsychiatry/fullarticle/2810311)

> Multicenter decision-analytic prediction model using CTN trial data.