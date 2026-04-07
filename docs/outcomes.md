# Outcomes

The pipeline includes 8 pre-defined clinical outcomes derived from the CTN-0094 dataset. All outcomes are based on urine drug screening (UDS) data collected across multiple weeks of treatment.

For full definitions, see the [CTNote package documentation](https://ctn-0094.github.io/CTNote/).

---

## Summary Table

| Outcome | Endpoint Type | Model Used | Description |
|:---|:---:|:---|:---|
| `ctn0094_relapse_event` | Logical | Logistic Regression | Any positive UDS during treatment |
| `Ab_krupitskyA_2011` | Logical | Logistic Regression | Confirmed abstinence weeks 5–24 |
| `Ab_ling_1998` | Logical | Logistic Regression | 13 consecutive negative UDS (~1 month) |
| `Rs_johnson_1992` | Logical | Logistic Regression | 2 consecutive positive UDS after 4-week treatment |
| `Rs_krupitsky_2004` | Logical | Logistic Regression | 3 consecutive positive UDS = relapse |
| `Rd_kostenB_1993` | Logical | Logistic Regression | 3 weeks of consecutive negative UDS |
| `Ab_schottenfeldB_2008` | Integer | Negative Binomial | Count of negative UDS weeks |
| `Ab_mokri_2016` | Survival | Cox Proportional Hazard | Time-to-first negative UDS |

---

## Binary Outcomes (Logical)

### `ctn0094_relapse_event`
**Any positive UDS event during the treatment period.**

The broadest relapse definition — flags any patient who had at least one positive drug screen. Used as the primary outcome for overall pipeline validation.

---

### `Ab_krupitskyA_2011`
**Confirmed abstinence across weeks 5–24.**

Based on Krupitsky et al. (2011). A patient is classified as abstinent if all available UDS results in weeks 5–24 are negative. Patients with missing screens during this window are classified as non-abstinent.

---

### `Ab_ling_1998`
**13 consecutive negative UDS results (~1 month of abstinence).**

Based on Ling et al. (1998). Requires a sustained run of clean screens — a stricter definition than `ctn0094_relapse_event`.

---

### `Rs_johnson_1992`
**2 consecutive positive UDS after the first 4 weeks of treatment.**

Based on Johnson et al. (1992). Focuses on relapse occurring after an initial stabilization period, distinguishing early relapse from treatment-period relapse.

---

### `Rs_krupitsky_2004`
**3 consecutive positive UDS = relapse event.**

Based on Krupitsky et al. (2004). Requires sustained positive screens before flagging relapse — less sensitive to single-screen noise.

---

### `Rd_kostenB_1993`
**3 weeks of consecutive negative UDS.**

Based on Kosten et al. (1993). A remission definition rather than a relapse definition — models the probability of achieving a sustained clean period.

---

## Count Outcome (Integer)

### `Ab_schottenfeldB_2008`
**Total count of negative UDS weeks during treatment.**

Based on Schottenfeld et al. (2008). Models the number of abstinent weeks rather than a binary event, capturing gradual improvement rather than just endpoint status. Uses Negative Binomial regression to account for overdispersion in count data.

---

## Survival Outcome

### `Ab_mokri_2016`
**Time-to-first negative UDS (days).**

Based on Mokri et al. (2016). Treats the first clean drug screen as an event and models the time it takes to reach that milestone. Uses Cox Proportional Hazard regression. Patients who never achieve a negative screen are treated as censored.

---

## Using Custom Outcomes

You can supply a custom outcome not in the list above using the `--type` flag:

```bash
python3 run_pipelineV2.py \
  --data data.csv \
  --outcome my_custom_outcome \
  --type logical \
  -d ./results
```

!!! warning "Dataset compatibility"
    The pipeline is built around the CTN-0094 data schema. Bringing in an external dataset is not plug-and-play — PSM requires specific columns (`RaceEth`, `age`, `is_female`) to be present and correctly formatted. Your dataset must also already contain a column matching the `--outcome` name, formatted correctly for the specified `--type`. Using the pipeline on an incompatible dataset will fail at the PSM step without a schema migration.
