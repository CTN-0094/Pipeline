# Input Format

The pipeline accepts a single cleaned CSV passed via `--data`. This page documents every column the preprocessing and PSM stages expect.

---

## Minimum Required Columns

These columns must always be present regardless of outcome type.

| Column | Type | Description |
|:---|:---|:---|
| `who` | integer | Unique patient identifier — hardcoded as the ID column throughout the pipeline |
| `RaceEth` | string | Demographic group label used for PSM splitting — see [RaceEth values](#raceeth) below |
| `age` | integer | Patient age in years — used as a PSM matching covariate by default |
| `Sex` | string | `"male"` or `"female"` — renamed to `is_female` (0/1) during preprocessing |

---

## Outcome Columns

At least one outcome column is required. Its type must match the `--type` or the built-in outcome config.

=== "Logical (binary)"

    | Column | Type | Valid values |
    |:---|:---|:---|
    | `<outcome_name>` | integer | `0` or `1` only |

=== "Integer (count)"

    | Column | Type | Valid values |
    |:---|:---|:---|
    | `<outcome_name>` | integer | Non-negative whole numbers |

=== "Survival (time-to-event)"

    Two columns are required. By convention the pipeline expects `columnsToUse = [time_col, event_col]`.

    | Column | Type | Valid values |
    |:---|:---|:---|
    | `<outcome_name>_time` | float | Positive duration (days, weeks, etc.) |
    | `<outcome_name>_event` | integer | `0` (censored) or `1` (event observed) |

---

## Demographic & Socioeconomic Columns

Used in feature engineering and model training. Missing columns are silently skipped by the preprocessor.

| Column | Type | Valid values | Notes |
|:---|:---|:---|:---|
| `education` | string | `"Less than HS"`, `"HS/GED"`, `"More than HS"` | Encoded 1–3; NaN → 0 |
| `marital` | string | `"Never married"`, `"Married or Partnered"`, `"Separated/Divorced/Widowed"` | Encoded 2–4; NaN → 1 |
| `job` | string | `"Full Time"`, `"Part Time"`, or any other value | Anything other than Full/Part Time → `unemployed = 1` |
| `is_living_stable` | integer | `0` or `1` | Inverted to `unstableliving` (1 = unstable) |
| `pain` | string | `"No Pain"`, `"Severe Pain"`, `"Very mild to Moderate Pain"`, `"Missing"` | Encoded to binary (0 = no pain) |
| `race` | string | `"White"`, `"Black"`, `"Other"`, `"Refused/missing"` | Encoded 1–3; dropped after encoding |

---

## Treatment Column

| Column | Type | Valid values |
|:---|:---|:---|
| `XTRT` | string | `"CTN30BUP"`, `"CTN51BUP"`, `"CTN51NTX"`, `"CTN27BUP"`, `"CTN27MET"` |

Encoded as integers 1–5. Rows with unrecognised values are set to `-1`.

---

## TLFB Columns

Timeline Follow-Back (TLFB) columns record the number of days each substance was used in a recall period.

The nine columns below are kept individually. Any other `TLFB_*` column present in the data is summed into a single `TLFB_Other` feature.

| Column | Substance |
|:---|:---|
| `TLFB_Alcohol_Count` | Alcohol |
| `TLFB_Amphetamine_Count` | Amphetamine |
| `TLFB_Cocaine_Count` | Cocaine |
| `TLFB_Heroin_Count` | Heroin |
| `TLFB_Benzodiazepine_Count` | Benzodiazepine |
| `TLFB_Opioid_Count` | Opioid (non-heroin) |
| `TLFB_THC_Count` | Cannabis |
| `TLFB_Methadone_Count` | Methadone |
| `TLFB_Buprenorphine_Count` | Buprenorphine |

---

## UDS Columns

Urine Drug Screening (UDS) count columns are binarised during preprocessing (any count > 0 → 1). Any column whose name starts with `UDS_` is processed this way.

```
UDS_Cocaine_Count, UDS_Opioid_Count, UDS_THC_Count, ...
```

!!! note
    `UDS_Alcohol_Count` and `UDS_Mdma/Hallucinogen_Count` are **dropped** before binarisation if present.

---

## Behavioural Columns

These two columns are used to compute derived binary features and are then dropped.

| Column | Type | Used to compute |
|:---|:---|:---|
| `msm_npt` | float | `Homosexual_Behavior` — 1 if `msm_npt > 0` and `Sex == "male"` |
| `txx_prt` | float | `Non_monogamous_Relationships` — 1 if `txx_prt > 1` |

---

## Other Columns

| Column | Type | Notes |
|:---|:---|:---|
| `heroin_inject_days` | float | Converted to binary `rbsivheroin` (1 if non-null, 0 if null) |
| `ftnd` | float | Fagerström score — NaN filled with 0, used as-is |
| `is_hispanic` | any | Dropped unconditionally before PSM |

---

## Columns Dropped During Preprocessing

The following columns are removed if present. The pipeline will not raise if they are absent.

```
pain_when, is_smoker, per_day, max, amount, depression, anxiety, schizophrenia,
cocaine_inject_days, speedball_inject_days, opioid_inject_days, speed_inject_days,
UDS_Alcohol_Count, UDS_Mdma/Hallucinogen_Count, msm_frq, msm_npt, txx_prt,
rbs_iv_days, race, RBS_cocaine_Days, RBS_heroin_Days, RBS_opioid_Days,
RBS_speed_Days, RBS_speedball_Days
```

---

## `RaceEth` Values { #raceeth }

`RaceEth` is the column used for PSM splitting. It must contain string labels that map to the following numeric codes during preprocessing:

| Label | Code | Group |
|:---|:---:|:---|
| `"NHW"` | 1 | Non-Hispanic White — default majority group |
| `"NHB"` | 2 | Non-Hispanic Black |
| `"Hisp"` | 3 | Hispanic |
| `"Other"` | 4 | Other |
| `"Refused/Missing"` | 0 | Refused or missing |

The default `--majority 1` targets NHW as the majority group. To use a different majority, pass `--majority <code>` with the numeric code from the table above.

---

## Using Custom Data

The pipeline is not restricted to CTN-0094 data. To run it on your own dataset:

1. Ensure your CSV has at minimum `who`, `RaceEth`, `age`, and `Sex` columns.
2. Add at least one outcome column.
3. Pass `--type` if your outcome is not one of the [pre-defined outcomes](outcomes.md).

```bash
python3 run_pipelineV2.py \
  --data my_data.csv \
  --outcome my_outcome \
  --type logical \
  --split RaceEth \
  --match "age is_female" \
  -d ./results
```

!!! tip "Minimal example"
    For a working minimal CSV, see the [smoke test data generator](https://github.com/CTN-0094/Pipeline/blob/main/run_pipelineV2.py) used during development — it constructs a 500-row synthetic dataset with all required columns.
