# References

## Pipeline Paper

<div class="paper-card" markdown>

### From Clinical Trials to Real-World Impact: Introducing a Computational Framework to Detect Endpoint Bias in Opioid Use Disorder Research

**Odom GJ, Brandt L, Marker A, Giorgi S, Jainarain G, Schwartz HA, Au L, Castro C, and the ENDPOINT Consortium.**
*Drug and Alcohol Review.* 2026;45(1):e70085.

[View Paper :material-open-in-new:](https://onlinelibrary.wiley.com/doi/10.1111/dar.70085){ .md-button .md-button--primary }

</div>

This is the primary publication describing the DAB Pipeline. It introduces the computational framework for detecting endpoint bias in OUD clinical trial data, demonstrates the pipeline across the CTN-0094 dataset, and presents the measurement invariance approach used to evaluate algorithmic fairness across demographic groups.

---

## Foundational Paper

The DAB Pipeline is a direct extension of the following publication. The pipeline replicates and generalizes the modeling approach from this paper, applying it across a broader set of outcomes and demographic configurations to study algorithmic bias.

---

<div class="paper-card" markdown>

### Individual-Level Risk Prediction of Return to Use During Opioid Use Disorder Treatment

**Luo SX, Feaster DJ, Liu Y, et al.**
*JAMA Psychiatry.* 2024;81(1):45–56.

[View Paper :material-open-in-new:](https://jamanetwork.com/journals/jamapsychiatry/fullarticle/2810311){ .md-button .md-button--primary }

</div>

#### What the paper did

Using data from four CTN trials (the same CTN-0094 dataset this pipeline is built on), Luo et al. developed and validated a multicenter decision-analytic model to predict individual-level return to opioid use during treatment. The model was trained on patient demographics and early treatment response features, and evaluated on a held-out test set.

#### What this pipeline extends

The DAB Pipeline takes the same modeling approach and asks a further question: **does the model perform equally well across different demographic groups?**

Specifically, the pipeline:

- Replicates the core prediction task (return to use, abstinence, relapse events) across 8 outcome definitions derived from the same dataset
- Introduces Propensity Score Matching to construct training cohorts with systematically varied majority/minority demographic compositions
- Evaluates whether model performance is stable across those compositions — i.e., whether the outcomes are *measurement invariant*

If a model's performance degrades as the minority proportion of the training cohort increases, that is evidence of algorithmic bias in the original modeling approach.

---

## Related Work

**Odom GJ et al.** Measurement invariance framework for evaluating algorithmic bias in clinical prediction models. *In preparation.* 2025.

> Provides the theoretical basis for the measurement invariance interpretation used in this pipeline.

---

## Citing This Pipeline

If you use the DAB Pipeline in your research, please cite both the foundational paper and this pipeline:

```bibtex
@article{odom2026dar,
  author  = {Odom, Gabriel J. and Brandt, Laura and Marker, Aaron and Giorgi, Salvatore
             and Jainarain, Ganesh and Schwartz, H. Andrew and Au, Larry and Castro, Clinton
             and {the ENDPOINT Consortium}},
  title   = {From Clinical Trials to Real-World Impact: Introducing a Computational
             Framework to Detect Endpoint Bias in Opioid Use Disorder Research},
  journal = {Drug and Alcohol Review},
  year    = {2026},
  volume  = {45},
  number  = {1},
  pages   = {e70085},
  doi     = {10.1111/dar.70085}
}

@article{luo2024jama,
  author  = {Luo, Sean X. and Feaster, Daniel J. and Liu, Yan and others},
  title   = {Individual-Level Risk Prediction of Return to Use During Opioid Use Disorder Treatment},
  journal = {JAMA Psychiatry},
  year    = {2024},
  volume  = {81},
  number  = {1},
  pages   = {45--56},
  doi     = {10.1001/jamapsychiatry.2023.3596}
}

@software{jainarain2026dab,
  author    = {Jainarain, Ganesh and Marker, Aaron and Odom, Gabriel J. and Brandt, Laura},
  title     = {DAB Pipeline: Detecting Algorithmic Bias in OUD Treatment Data},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/CTN-0094/Pipeline}
}
```
