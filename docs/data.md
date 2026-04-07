# Data

All datasets used in the DAB Pipeline are publicly available on GitHub. They are de-identified and derived from the [CTN-0094](https://ctnlibrary.org/protocol/ctn0094/) clinical trial.

---

## Datasets

### Master Dataset
The primary cleaned dataset used as input to the pipeline. Contains patient demographics, treatment features, and TLFB-derived variables.

- **File:** `master_data_clean.csv`
- **Size:** ~302 KB

[Download :material-download:](https://github.com/CTN-0094/Pipeline/raw/main/data/master_data_clean.csv){ .md-button .md-button--primary }

---

### Outcomes Dataset
Merged dataset containing all pre-defined outcome columns alongside patient features.

- **File:** `outcomes_merged_dataset.csv`
- **Size:** ~674 KB

[Download :material-download:](https://github.com/CTN-0094/Pipeline/raw/main/data/outcomes_merged_dataset.csv){ .md-button .md-button--primary }

---

### Held-Out Test Set
The stratified held-out evaluation set (58% majority / 42% minority) used for final model evaluation.

- **File:** `test_holdout.csv`
- **Size:** ~14 KB

[Download :material-download:](https://github.com/CTN-0094/Pipeline/raw/main/data/test_holdout.csv){ .md-button .md-button--primary }

---

### Training Pool
The remaining data after the held-out set is removed. Used as the source for PSM cohort construction.

- **File:** `train_pool_after_stratified_test_removal.csv`
- **Size:** ~294 KB

[Download :material-download:](https://github.com/CTN-0094/Pipeline/raw/main/data/train_pool_after_stratified_test_removal.csv){ .md-button .md-button--primary }

---

### Ratio Sets Bundle
Pre-computed PSM cohorts across all 11 majority/minority demographic ratios, bundled as a zip archive.

- **File:** `ratio_sets_bundle.zip`
- **Size:** ~136 KB

[Download :material-download:](https://github.com/CTN-0094/Pipeline/raw/main/data/ratio_sets_bundle.zip){ .md-button .md-button--primary }

---

## Data Source

The CTN-0094 dataset is a harmonized, de-identified resource compiled from four NIDA Clinical Trials Network (CTN) studies of opioid use disorder treatment:

| CTN Trial | Treatment | N |
|:---|:---|:---:|
| CTN-0027 | Buprenorphine taper | 653 |
| CTN-0028 | Buprenorphine taper + naltrexone | 653 |
| CTN-0030 | Buprenorphine extended taper | 570 |
| CTN-0051 | Extended-release naltrexone vs. buprenorphine | 570 |

For full documentation of the source trials and variable definitions, see the [CTNote package](https://ctn-0094.github.io/CTNote/).

!!! info "Citation"
    If you use these datasets in your research, please cite the original CTN-0094 harmonization work alongside this pipeline.
