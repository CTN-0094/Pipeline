Using the Pipeline: A Step-by-Step Walkthrough
===============================================

This page walks through a full end-to-end run of the CTN-0094 Pipeline — from setup to interpreting results — in a Colab notebook style.
You can follow along locally or click the badge below to run it live in Google Colab.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1Agj-EXE9WhLjMulA1mIVzW4YWNN7fiqX
   :alt: Open In Colab

----

Setup
-----

**Cell 1 — Clone the repository**

.. code-block:: python

    import os

    if not os.path.exists("/content/Pipeline"):
        !git clone https://github.com/CTN-0094/Pipeline.git

    os.chdir("/content/Pipeline")

.. code-block:: none
   :class: cell-output

    Cloning into 'Pipeline'...
    remote: Enumerating objects: 412, done.
    remote: Counting objects: 100% (412/412), done.
    Resolving deltas: 100% (231/231), done.

----

**Cell 2 — Install dependencies**

This installs all Python packages required by the pipeline.

.. code-block:: python

    !pip install -r requirements.txt -q

.. code-block:: none
   :class: cell-output

    Successfully installed lifelines-0.29.0 psmpy-0.3.13 statsmodels-0.14.1 ...

----

**Cell 3 — Verify the install**

.. code-block:: python

    !python -c "from src.train_model import LogisticModel; print('Import OK')"

.. code-block:: none
   :class: cell-output

    Import OK

----

Running the Pipeline
--------------------

**Cell 4 — Run with a single outcome**

The minimum required argument is ``--data``, pointing to your cleaned CSV file, and ``--outcome`` specifying which endpoint to model.

.. code-block:: python

    !python run_pipelineV2.py \
        --data /content/my_data.csv \
        --outcome ctn0094_relapse_event \
        -d /content/results

.. code-block:: none
   :class: cell-output

    Processing subset 1 ____________________________________________________________
    Lasso feature selection completed. Selected 8 out of 24 features.
    Features are:  ['age', 'is_female', 'RaceEth', 'UDS_Opioid_Count', ...]

    Processing subset 2 ____________________________________________________________
    Lasso feature selection completed. Selected 6 out of 24 features.
    ...
    ________________________________________________________________________

    Elapsed time: 12.43 seconds

----

**Cell 5 — Run across multiple seeds**

Use ``-l <min> <max>`` to loop through a range of random seeds. Each seed produces a different PSM cohort, letting you measure stability across sampling variation.

.. code-block:: python

    !python run_pipelineV2.py \
        --data /content/my_data.csv \
        --outcome ctn0094_relapse_event \
        -l 1 10 \
        -d /content/results

.. note::
   This will run the full pipeline 9 times (seeds 1 through 9) and save a separate results folder for each seed under ``/content/results``.

----

**Cell 6 — Run multiple outcomes at once**

Combine ``-l`` and ``-o`` to sweep over several outcomes and several seeds in one call.

.. code-block:: python

    !python run_pipelineV2.py \
        --data /content/my_data.csv \
        -l 0 5 \
        -o ctn0094_relapse_event Ab_ling_1998 Rs_johnson_1992 \
        -d /content/results

.. code-block:: none
   :class: cell-output

    Running outcome: ctn0094_relapse_event | seed 0
    Running outcome: ctn0094_relapse_event | seed 1
    ...
    Running outcome: Rs_johnson_1992 | seed 4

----

**Cell 7 — Preprocess and match only (no model training)**

Use ``--data_only`` to stop after propensity score matching and save the ML-ready datasets without training any models. Useful for inspecting cohort balance before committing to a full run.

.. code-block:: python

    !python run_pipelineV2.py \
        --data /content/my_data.csv \
        --outcome ctn0094_relapse_event \
        --data_only \
        -d /content/psm_output

.. code-block:: none
   :class: cell-output

    Preprocessing complete. PSM subsets saved to /content/psm_output/

----

Customising the Cohort
----------------------

**Cell 8 — Change cohort size and held-out composition**

The pipeline defaults to groups of 500 matched participants and a held-out set of 100 (58% majority). You can override both.

.. code-block:: python

    !python run_pipelineV2.py \
        --data /content/my_data.csv \
        --outcome ctn0094_relapse_event \
        --group_size 300 \
        --heldout_size 80 \
        --heldout_set_percent_majority 50 \
        -d /content/results

----

**Cell 9 — Change the matching column or matching covariates**

By default the pipeline splits on ``RaceEth`` and matches on ``age`` and ``is_female``. Change these with ``--split`` and ``--match``.

.. code-block:: python

    !python run_pipelineV2.py \
        --data /content/my_data.csv \
        --outcome ctn0094_relapse_event \
        --split RaceEth \
        --match "age is_female UDS_Opioid_Count" \
        -d /content/results

----

Using a Custom Outcome
----------------------

**Cell 10 — Bring your own outcome column**

If your dataset contains a column not in the built-in outcome list, pass its name with ``-o`` and specify the endpoint type with ``--type``.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``--type``
     - When to use
   * - ``logical``
     - Binary outcome (0/1)
   * - ``integer``
     - Count outcome (0, 1, 2, …)
   * - ``survival``
     - Time-to-event; expects two columns: ``<name>_time`` and ``<name>_event``

.. code-block:: python

    # Binary custom outcome
    !python run_pipelineV2.py \
        --data /content/my_data.csv \
        --outcome my_custom_outcome \
        --type logical \
        -d /content/results

.. code-block:: python

    # Survival custom outcome
    # Expects columns: my_event_time  and  my_event_event
    !python run_pipelineV2.py \
        --data /content/my_data.csv \
        --outcome my_event \
        --type survival \
        -d /content/results

----

Viewing the Results
-------------------

**Cell 11 — List output files**

.. code-block:: python

    import os

    for root, dirs, files in os.walk("/content/results"):
        level = root.replace("/content/results", "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"  {indent}{f}")

.. code-block:: none
   :class: cell-output

    results/
      ctn0094_relapse_event_seed0/
        evaluations/
          evaluations_2025-01-15.csv
        predictions/
          predictions_2025-01-15.csv
        logs/
          pipeline_2025-01-15.log

----

**Cell 12 — Load and inspect evaluations**

.. code-block:: python

    import pandas as pd, glob

    files = glob.glob("/content/results/**/evaluations/*.csv", recursive=True)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.head()

.. code-block:: none
   :class: cell-output

    ╔══════════════╦══════════╦══════════╦══════════╦═══════════════════════╗
    ║  Data Type   ║  roc     ║ precision║  recall  ║  training_demographics║
    ╠══════════════╬══════════╬══════════╬══════════╬═══════════════════════╣
    ║  heldout     ║  0.71    ║  0.68    ║  0.74    ║  250 White, 250 Black ║
    ║  subset      ║  0.69    ║  0.65    ║  0.71    ║  250 White, 250 Black ║
    ╚══════════════╩══════════╩══════════╩══════════╩═══════════════════════╝

----

**Cell 13 — Plot ROC-AUC across seeds**

.. code-block:: python

    import matplotlib.pyplot as plt

    heldout = df[df.index % 2 == 0]["roc"].values   # heldout rows
    subset  = df[df.index % 2 == 1]["roc"].values   # subset rows

    plt.figure(figsize=(8, 4))
    plt.plot(heldout, marker="o", label="Held-out")
    plt.plot(subset,  marker="s", label="Subset")
    plt.axhline(0.5, linestyle="--", color="gray", label="Chance")
    plt.xlabel("Seed")
    plt.ylabel("ROC-AUC")
    plt.title("Model performance across PSM seeds")
    plt.legend()
    plt.tight_layout()
    plt.show()

----

Interpreting Results
--------------------

The pipeline is designed to detect **algorithmic bias** — a disparity in model performance when trained on cohorts with different demographic compositions.

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Result
     - Interpretation
   * - ROC-AUC stable across seeds
     - The outcome is **measurement invariant** — model performance does not depend on cohort demographics.
   * - ROC-AUC varies across seeds
     - The outcome is **measurement variant** — model performance is sensitive to demographic composition, which may indicate bias.
   * - Held-out AUC >> Subset AUC
     - Possible over-fitting to the matched cohort; consider increasing ``--group_size`` or reducing features.
   * - No features selected (Lasso error)
     - Regularisation may be too strong; the alpha defaults are tuned for the CTN-0094 dataset and may need adjustment for other data.

----

Available Outcomes Reference
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Outcome
     - Type
     - Description
   * - ``ctn0094_relapse_event``
     - Logical
     - Any positive urine drug screen
   * - ``Ab_krupitskyA_2011``
     - Logical
     - Confirmed abstinence (weeks 5–24)
   * - ``Ab_ling_1998``
     - Logical
     - 13 consecutive negative UDS
   * - ``Rs_johnson_1992``
     - Logical
     - 2 consecutive positive UDS after 4-week treatment
   * - ``Rs_krupitsky_2004``
     - Logical
     - 3 consecutive positive UDS
   * - ``Rd_kostenB_1993``
     - Logical
     - 3 consecutive negative UDS
   * - ``Ab_schottenfeldB_2008``
     - Integer
     - Count-based abstinence measure
   * - ``Ab_mokri_2016``
     - Survival
     - Time-to-event abstinence outcome
