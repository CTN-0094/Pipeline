# Changelog

All notable changes to the CTN-0094 Pipeline project are documented here.

---

## [Unreleased]

### Added
- `app.py` — Gradio web interface for running the pipeline and visualising results interactively (metrics, ROC AUC, confusion matrix, and bias plots)
- `tests/test_preprocess_pipeline.py` — integration-level tests for the `preprocess_data()` orchestrator

---

## [0.5.0] — 2026-05

### Added
- `RACEETH_LABELS` constant mapping numeric demographic codes to readable labels
- Input format reference page in MkDocs site (`docs/input_format.md`)
- API reference page wired into the site navigation

### Changed
- Replaced `print()` calls with `logging` throughout; removed separator lines from log output
- Logging now uses dual handlers (file + console) with color-coded severity levels

---

## [0.4.0] — 2025-12

### Added
- Full MkDocs Material site replacing the previous Sphinx documentation
- Site pages: quickstart, pipeline walkthrough, outcomes, data format, API reference, testing, and references
- Code reference section auto-generated from docstrings
- `CITATION.cff` and MIT `LICENSE` for academic attribution
- GitHub issue templates for bug reports and feature requests
- Odom et al. 2026 (*Drug and Alcohol Review*) added to the references page

### Changed
- `pyproject.toml` updated; redundant `setup.py` removed

### Fixed
- NIH badge link corrected to point to the CTN-0094 protocol page
- Broken Google Colab link in the beginner tutorial
- Code cell readability: dark background with light text in rendered docs
- Stale API signatures updated in documentation

---

## [0.3.0] — 2025-10

### Added
- `src/validate.py` — dataset validation for logical, integer, and survival endpoint types
- `src/constants.py` — `EndpointType` enum and shared pipeline constants
- Input validation added to `OutcomeModel.__init__()` (empty DataFrame, wrong column types, missing columns, duplicate IDs)
- Unit test suite under `tests/`:
  - `test_constants.py` — enum lookup and value checks
  - `test_data_ingestion.py` — CSV loading and data structure
  - `test_holdout_split.py` — split mechanics, demographic ratios, data integrity, edge cases
  - `test_input_validation.py` — `OutcomeModel` constructor guards
  - `test_model_training.py` — model selection, results structure, subset processing
  - `test_preprocess.py` — all ten `DataPreprocessor` public methods
  - `test_train_model.py` — logistic model feature selection, training, and evaluation
  - `test_validate.py` — all `validate_dataset_for_model` code paths
- Legacy unit tests under `unitTests/`: `test_model_fit_and_eval.py`, `test_psm.py`

### Fixed
- Patched 7 dependency vulnerabilities across 5 packages

---

## [0.2.0] — 2025-06

### Added
- `src/logging_setup.py` — centralised logging configuration
- `src/silent_logging.py` — suppresses third-party library noise during runs
- NBR (negative binomial regression) and survival model stubs (`src/train_model.py`)
- `--data`, `--outcome`, `--group_size`, and `--data_only` command-line arguments to `run_pipelineV2.py`
- `--data_only` flag saves demographic subsets without training models

### Changed
- `run_pipelineV2.py` refactored: preprocessing, KNN/PSM subset creation, and model training moved into separate source modules
- Terminal output cleaned up; redundant print statements removed

### Fixed
- Off-by-one error in evaluation results indexing
- `who` column correctly excluded from training/test features while being tracked for evaluation
- Seed handling: specifying a seed list no longer raises an error
- Log scraper now dynamically parses any new pipeline run

---

## [0.1.0] — 2025-01

### Added
- Initial pipeline: `run_pipelineV2.py` orchestrating propensity-score matching and logistic regression for `ctn0094_relapse_event`
- `src/preprocess.py` (`DataPreprocessor`) — column dropping, Yes/No binarisation, TLFB aggregation, UDS binarisation, categorical encoding, and column renaming
- `src/preprocess_pipeline.py` — thin orchestrator wrapping `DataPreprocessor` steps
- `src/create_demodf_knn.py` — holdout split, propensity score matching via `rpy2`/`MatchIt`, and 11-subset creation
- `src/model_training.py` — `OutcomeModel` class with logistic regression, feature selection (VIF), and heldout evaluation
- `src/train_model.py` — `LogisticModel` wrapping `sklearn` logistic regression with AUC, confusion matrix, and fairness metrics
- `src/data_preprocessing.py` — thin wrapper logging around `preprocess_data()`
- `src/utils.py` — pipeline completion utility
- `data/outcomes_merged_dataset.csv` and supporting data files
- `README.md` with quick-start instructions and project description
- `.gitignore` excluding virtualenv, `__pycache__`, and generated artefacts
- `requirements.txt` and `pyproject.toml` with all dependencies
- GitHub Actions workflow for MkDocs deployment to GitHub Pages
