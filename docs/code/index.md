# Code Reference

The pipeline is split into focused modules under `src/`. Each page below documents the public functions and classes in that module.

| Module | File | Description |
|:---|:---|:---|
| [Constants](constants.md) | `constants.py` | `EndpointType` enum |
| [Validation](validate.md) | `validate.py` | Pre-training dataset validation |
| [Preprocessing](preprocess.md) | `preprocess.py` | Feature engineering and data cleaning |
| [PSM & Cohorts](psm.md) | `create_demodf_knn.py` | Propensity score matching and subset construction |
| [Model Training](model_training.md) | `model_training.py` | Training orchestration across subsets |
| [Models](models.md) | `train_model.py` | Model class implementations |
