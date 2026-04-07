# DAB Pipeline

<div class="hero" markdown>

**A modular, reproducible pipeline for detecting algorithmic bias in opioid use disorder treatment data.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Agj-EXE9WhLjMulA1mIVzW4YWNN7fiqX)
&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/CTN-0094/Pipeline/blob/main/LICENSE)
&nbsp;
[![NIH AIM-AHEAD](https://img.shields.io/badge/NIH-AIM--AHEAD-green)](https://ctnlibrary.org/protocol/ctn0094/)

</div>

---

## What is the DAB Pipeline?

The **Detecting Algorithmic Bias (DAB) Pipeline** evaluates whether machine learning models for predicting opioid use disorder (OUD) treatment outcomes perform consistently across demographic groups. It uses the [CTN-0094](https://ctn-0094.github.io/Pipeline/) clinical trial dataset and supports multiple modeling strategies and outcome definitions.

We define **algorithmic bias** as a disparity in model performance between different demographic groups — for example, a model that predicts relapse accurately for White patients but not for Black or Hispanic patients.

---

## Quick Start

```bash
python3 run_pipelineV2.py --data data.csv --outcome ctn0094_relapse_event -d ./results
```

[Get started :material-arrow-right:](quickstart.md){ .md-button .md-button--primary }
[How it works :material-arrow-right:](pipeline.md){ .md-button }

---

## Key Features

<div class="grid cards" markdown>

-   :material-scale-balance:{ .lg .middle } **Fairness-First Design**

    ---

    Every evaluation includes the demographic composition of the training cohort, making bias auditing a first-class output.

-   :material-dna:{ .lg .middle } **Clinical Outcomes**

    ---

    8 pre-defined OUD outcomes across binary, count, and time-to-event endpoint types, derived from the CTN-0094 trial.

-   :material-swap-horizontal:{ .lg .middle } **Modular Architecture**

    ---

    Swap out models or outcome definitions without touching the core pipeline logic. External datasets must conform to the CTN-0094 schema required by PSM.

-   :material-chart-line:{ .lg .middle } **Propensity Score Matching**

    ---

    Constructs balanced cohorts via PSM across a spectrum of majority/minority demographic ratios.

</div>

---

## Built On

This pipeline extends the work of Luo et al. (2024), replicating their OUD return-to-use prediction model and adding a fairness analysis layer to evaluate how model performance shifts across demographic compositions.

[Read more :material-arrow-right:](references.md){ .md-button }

---

## Team

| Role | Name | Institution |
|:---|:---|:---|
| Clinical Lead | Prof. Laura Brandt | City College of New York |
| Computational Lead | Prof. Gabriel Odom | Florida International University |
| Data Scientist | Ganesh Jainarain | City College of New York |
| Data Scientist | Aaron Marker | Stony Brook University |
| Funding | NIH AIM-AHEAD | `1OT2OD032581-02-267` |

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{jainarain2026dab,
  author    = {Jainarain, Ganesh and Marker, Aaron and Odom, Gabriel J. and Brandt, Laura},
  title     = {DAB Pipeline: Detecting Algorithmic Bias in OUD Treatment Data},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/CTN-0094/Pipeline}
}
```
