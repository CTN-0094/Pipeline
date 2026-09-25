# How It Works

The DAB pipeline runs five main stages for each outcome and seed combination. They are mostly sequential, with two branches: the held-out set splits off before matching, and `--data_only` stops the run after cohort construction.

---

## Pipeline Overview

```mermaid
flowchart TD
    IN[("Input CSV<br/>--data")] --> VAL{"validate_dataset_for_model"}
    VAL -->|fails| EXIT(["SystemExit"])
    VAL -->|passes| PRE["preprocess_merged_data<br/>encoding · TLFB features"]
    PRE --> DROP["drop_other_outcomes<br/>keep only this run's endpoint"]

    DROP --> SEED["seed set · random + numpy"]
    SEED --> HOLD["holdOutTestData"]

    HOLD --> HELD[/"Held-out set<br/>--heldout_size · stratified"/]
    HOLD --> POOL[/"Training pool"/]

    POOL --> PSM["propensityScoreMatch<br/>--split · --match · --group_size"]
    PSM --> SUB["create_subsets<br/>11 cohorts · 0/1000 → 500/500 ladder"]

    SUB --> ONLY{"--data_only?"}
    ONLY -->|yes| SAVE["save_model_input_datasets"]
    SAVE --> DONE(["done"])

    ONLY -->|no| DISP{"endpointType"}
    DISP -->|logical| LOG["LogisticModel"]
    DISP -->|integer| NB["NegativeBinomialModel"]
    DISP -->|survival| COX["CoxProportionalHazard"]

    LOG & NB & COX --> FIT["per subset × 11<br/>selectFeatures → train → evaluate"]
    HELD --> FIT

    FIT --> SP[/"subset_predictions"/]
    FIT --> HP[/"heldout_predictions"/]
    FIT --> SE[/"subset_evaluations"/]
    FIT --> HE[/"heldout_evaluations"/]

    SP & HP & SE & HE --> LOOP(["repeat over seeds × outcomes"])

    classDef gate fill:none,stroke:#d97706,stroke-width:1.5px;
    classDef data fill:none,stroke:#0891b2,stroke-width:1.5px;
    classDef term fill:none,stroke:#6b7280,stroke-width:1.5px;
    class VAL,ONLY,DISP gate;
    class HELD,POOL,SP,HP,SE,HE data;
    class EXIT,DONE,LOOP term;
```

Every path above runs once per `(outcome, seed)` pair. Two edges are easy to miss in a
step-by-step reading:

!!! info "Structure worth noting"
    The **held-out set is carved out before matching**, so it never passes through PSM or the
    subset ladder and stays untouched until evaluation. And **`--data_only` exits after cohort
    construction**, before any model is fit.

---

## Step 1 — Validation & Preprocessing

The pipeline validates the input CSV against expected schemas before any modeling occurs. It checks:

- Required columns are present
- Target column values match the endpoint type (binary 0/1, non-negative integer, positive duration)
- No duplicate patient IDs

Feature engineering includes binary encoding of categorical variables and construction of Treatment Lifecycle Feedback (TLFB) features from weekly urine drug screening data.

---

## Step 2 — Propensity Score Matching (PSM)

PSM builds a ladder of training cohorts whose demographic composition changes
while everything else about them stays as comparable as possible. It runs in two
parts: matching, then the ladder itself.

### Part 1 — Matching

Every row is flagged as minority or majority by the `--split` column: majority
means `RaceEth == 1` (Non-Hispanic White, by default `--majority 1`), and every
other value is minority.

The data is handed to R's [`MatchIt`](https://kosukeimai.github.io/MatchIt/) via
`rpy2`, which fits a probit-link GLM propensity model on the `--match` covariates
(`age` and `is_female` by default) and performs **optimal matching at a 1:2
ratio** — each minority participant is paired with two majority controls who look
like them on those covariates. Only the first `--group_size` (default 500)
minority participants are matched.

The result is **500 matched triples**, reshaped so that each row holds one
treated participant and their two controls:

| | treated_row | control_row_0 | control_row_1 |
|:--|:--|:--|:--|
| triple 1 | minority participant | majority control | majority control |
| triple 2 | minority participant | majority control | majority control |
| … | … | … | … |

Those three columns become three DataFrames of 500 rows each — `[minority,
majority_A, majority_B]` — still aligned row by row, so row *i* of all three
frames belongs to the same matched triple. That alignment is what makes the next
part work.

### Part 2 — The ladder

`create_subsets` builds 11 cohorts. Cohort *k* (counting from 0) takes:

- the **first `k × 50` rows** of the minority frame,
- the majority_A rows **from `k × 50` onward**,
- **all 500 rows** of majority_B.

So each step swaps 50 majority_A controls out for the 50 minority participants
they were matched to. Because the swap happens *inside* matched triples, the
cohort's age and sex profile stays balanced even as its racial composition
changes — which is the whole point. majority_B is never touched; it is the
constant backbone present in every cohort.

Total size stays fixed at **1000** on every rung:

| Cohort | Minority | Majority | Minority share |
|:--:|--:|--:|--:|
| 1 | 0 | 1000 | 0% |
| 2 | 50 | 950 | 5% |
| 3 | 100 | 900 | 10% |
| 4 | 150 | 850 | 15% |
| 5 | 200 | 800 | 20% |
| 6 | 250 | 750 | 25% |
| 7 | 300 | 700 | 30% |
| 8 | 350 | 650 | 35% |
| 9 | 400 | 600 | 40% |
| 10 | 450 | 550 | 45% |
| 11 | 500 | 500 | 50% |

Every model is trained once per cohort, so each run produces 11 sets of metrics
that can be read against the composition above.

!!! warning "The ladder tops out at 50/50"
    It does **not** sweep to an all-minority cohort. Because majority_B is always
    included in full, minority representation can never exceed half the cohort.
    The sweep is 0% → 50%, in 5-point steps.

!!! note "How the parameters interact"
    `--group_size` sets both the matched group size and the step: each cohort
    totals `2 × group_size`, and the step is `group_size ÷ 10`. The rung count
    (11) is fixed in `create_subsets` and is not exposed on the CLI. If
    `group_size` is not divisible by 10, the final rung falls slightly short of a
    balanced cohort — `--group_size 505` still steps by 50 and stops at 500
    minority rows.

### The held-out set

A stratified held-out evaluation set is carved out **before** any of this, so it
never passes through matching or the ladder. It holds a fixed majority/minority
ratio (`--heldout_set_percent_majority`, default 58%) so evaluation demographics
stay constant no matter how the training cohort is rebalanced. It is also sampled
with a module-local seed of 42 rather than the run seed, so the same rows are
held out across a multi-seed run.

---

## Step 3 — Feature Selection

L1 (Lasso) regularization is applied to automatically select predictive features before training. Features with zero coefficients after regularization are dropped. This reduces dimensionality and prevents overfitting on small cohorts.

!!! note
    If Lasso drops all features (over-regularization), the pipeline raises an error rather than silently training on no signal.

---

## Step 4 — Model Training

The model class is automatically selected based on the endpoint type of the chosen outcome:

| Endpoint Type | Model | Library |
|:---|:---|:---|
| `logical` — binary | Logistic Regression (L1) | scikit-learn |
| `integer` — count | Negative Binomial Regression | statsmodels |
| `survival` — time-to-event | Cox Proportional Hazard | lifelines |

Each model is trained independently on each PSM cohort.

---

## Step 5 — Evaluation

Each trained model is evaluated on two sets:

- **Internal test split** — 25% of the PSM cohort held out during training
- **Held-out set** — the fixed held-out set constructed in Step 2

Both evaluations record the demographic makeup of the training cohort alongside the metrics.

### Metrics by Endpoint Type

=== "Logical (binary)"

    | Metric | Description |
    |:---|:---|
    | ROC-AUC | Area under the receiver operating characteristic curve |
    | Precision | True positives / (true positives + false positives) |
    | Recall | True positives / (true positives + false negatives) |
    | Confusion Matrix | Full 2×2 breakdown |

=== "Integer (count)"

    | Metric | Description |
    |:---|:---|
    | MSE | Mean squared error |
    | RMSE | Root mean squared error |
    | MAE | Mean absolute error |
    | Pearson r | Linear correlation between predicted and actual |
    | McFadden R² | Goodness-of-fit relative to null model |

=== "Survival (time-to-event)"

    | Metric | Description |
    |:---|:---|
    | Concordance Index | C-statistic: probability model ranks a random pair correctly |

---

## Interpreting Results

The key question is: **does model performance change as the demographic composition of the training cohort changes?**

- **No change across ratios** → the outcome measure is *measurement invariant* — the model generalizes across groups equally
- **Performance drops as minority proportion increases** → the outcome is *measurement variant* — the model has learned patterns specific to the majority group

This framing is based on the measurement invariance framework from Odom et al. (2025).
