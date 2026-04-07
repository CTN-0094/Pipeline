# Getting Started

## Prerequisites

- Python 3.12+
- R (required for Propensity Score Matching via `MatchIt`)
- Git

---

## Installation

=== "Local"

    ```bash
    git clone https://github.com/CTN-0094/Pipeline.git
    cd Pipeline
    pip install -r requirements.txt
    ```

=== "Google Colab"

    ```python
    import os
    if not os.path.exists("/content/Pipeline"):
        !git clone https://github.com/CTN-0094/Pipeline.git
        os.chdir("/content/Pipeline")

    !pip install -r requirements.txt
    ```

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Agj-EXE9WhLjMulA1mIVzW4YWNN7fiqX)

---

## Running the Pipeline

### Basic Usage

```bash
python3 run_pipelineV2.py --data <path/to/data.csv> --outcome <outcome_name> -d ./results
```

### Examples

=== "Single outcome"

    ```bash
    python3 run_pipelineV2.py \
      --data data.csv \
      --outcome ctn0094_relapse_event \
      -d ./results
    ```

=== "Multi-seed loop"

    ```bash
    # Loop through seeds 5–10 across all outcomes
    python3 run_pipelineV2.py \
      --data data.csv \
      -l 5 10 \
      -d ./results
    ```

=== "Subset of outcomes"

    ```bash
    python3 run_pipelineV2.py \
      --data data.csv \
      -l 5 10 \
      -o ctn0094_relapse_event Ab_ling_1998 \
      -d ./results
    ```

=== "Data only (no training)"

    ```bash
    # Run preprocessing + PSM only, skip model training
    python3 run_pipelineV2.py \
      --data data.csv \
      --outcome ctn0094_relapse_event \
      --data_only \
      -d ./results
    ```

---

## CLI Reference

| Argument | Default | Description |
|:---|:---:|:---|
| `--data` | *(required)* | Path to cleaned input dataset (CSV) |
| `-o / --outcome` | all | Outcome(s) to run |
| `-l / --loop` | — | Min and max seed for multi-seed runs |
| `-d / --dir` | `""` | Output directory for results |
| `--type` | — | Endpoint type for custom outcomes: `logical`, `integer`, `survival` |
| `--majority` | `1` | Value representing the majority group in the PSM column |
| `--split` | `RaceEth` | Column used to define majority/minority groups |
| `--match` | `age is_female` | Columns to match on during PSM |
| `--group_size` | `500` | Size of each PSM group |
| `--heldout_size` | `100` | Size of the held-out evaluation set |
| `--heldout_set_percent_majority` | `58` | Percent majority in the held-out set |
| `--data_only` | `False` | Skip model training, save preprocessed data only |
| `-p / --prof` | — | Profiling mode: `simple` or `complex` |

---

## Viewing Results

After a run, your output directory contains:

```
results/
├── logs/                     # Execution logs per run
├── heldout_predictions/      # Held-out set predictions (CSV)
├── subset_predictions/       # Subset predictions by ratio (CSV)
├── heldout_evaluations/      # Metrics on the held-out set (CSV)
├── subset_evaluations/       # Metrics per demographic ratio (CSV)
└── experiments/              # Experiment metadata (JSON)
```

!!! tip "Reading the evaluation CSVs"
    Each row in an evaluation CSV corresponds to a different majority/minority demographic ratio. The rightmost columns are the evaluation metrics. Compare rows to see how model performance changes as the training cohort composition shifts.

---

## R Package Installation

When you run PSM for the first time, you will be prompted to install R packages:

```
Would you like to install MatchIt?
1: Yes
2: No

Selection: 1
```

Select `1` for both `MatchIt` and `optmatch`. This only happens once.
