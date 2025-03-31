Getting Started with the CTN-0094 Pipeline (Beginner Tutorial)
==============================================================

Welcome! This tutorial will walk you through everything you need to set up and run the CTN-0094 Pipeline on your computer — even if you've never used Python or Git before.

We’ll go step by step. Just follow along.

--------------------------------------------------

What You'll Need
----------------

Make sure you have:

1. A computer (Windows, macOS, or Linux)
2. An internet connection
3. About 15-20 minutes of focus

We’ll help you install:

- Python 3
- Git (used to download the project)
- A text editor like VS Code (optional but helpful)

--------------------------------------------------

Step 1: Install Python
----------------------

1. Go to https://www.python.org/downloads/
2. Click the yellow button to download Python for your system (Windows or macOS)
3. Install it

    **Important (Windows only):** When the installer opens, check the box that says:

    .. code-block::

       Add Python to PATH

    Then click “Install Now”

4. Once installed, open your terminal:

    - Windows: press `Win + R`, type `cmd`, hit Enter
    - macOS: open “Terminal” from Spotlight or Launchpad

5. Type this and press Enter:

    .. code-block:: bash

       python --version

    or (on Mac)

    .. code-block:: bash

       python3 --version

You should see something like:

.. code-block::

   Python 3.10.12

If it works — congrats! Python is installed.

--------------------------------------------------

Step 2: Install Git
-------------------

1. Go to https://git-scm.com/downloads
2. Download and install Git for your system
3. After installing, open your terminal and run:

    .. code-block:: bash

       git --version

You should see a version number like:

.. code-block::

   git version 2.42.0

--------------------------------------------------

Step 3: Download the Project
----------------------------

Let’s “clone” (download) the CTN-0094 Pipeline from GitHub.

1. In your terminal:

    .. code-block:: bash

       git clone https://github.com/CTN-0094/Pipeline.git

2. Move into the folder:

    .. code-block:: bash

       cd Pipeline

Now you’re inside the project!

--------------------------------------------------

Step 4: Set Up a Python Environment
-----------------------------------

This will create a special space just for this project so we don’t mess with your other files.

1. In your terminal:

    On **Windows**:

    .. code-block:: bash

       python -m venv venv
       venv\Scripts\activate

    On **macOS/Linux**:

    .. code-block:: bash

       python3 -m venv venv
       source venv/bin/activate

If you see something like `(venv)` at the start of your terminal line, it worked!

--------------------------------------------------

Step 5: Install Project Requirements
------------------------------------

Now we’ll install the tools and libraries the pipeline needs.

1. Inside your terminal (with venv activated):

    .. code-block:: bash

       pip install pandas numpy scikit-learn matplotlib seaborn lifelines joblib xgboost

That’s it! You’re ready to run the pipeline.

--------------------------------------------------

Step 6: Run the Pipeline
------------------------

The file that runs everything is called `run_pipelineV2.py`

1. To check it’s working, type:

    .. code-block:: python

       python run_pipelineV2.py --help

This will print all the options you can use.

2. To run a basic example:

    .. code-block:: python

       python run_pipelineV2.py -d results -s 123 -o opioid_use_past_30days

This command means:

- `-d results` — save files to a folder named “results”
- `-s 123` — use 123 as the random seed
- `-o opioid_use_past_30days` — use this outcome as the target

You’ll see logs appear as the pipeline runs!

--------------------------------------------------

Step 7: What Happens After
--------------------------

After the run is finished, you’ll have new folders:

- `results/logs/` — contains logs
- `results/predictions/` — your model predictions
- `results/evaluations/` — accuracy, AUC, and other metrics

You can open the files with Excel or a text editor.

--------------------------------------------------

(Optional) Step 8: Save Your Environment
---------------------------------------

If you want to save the environment you used for sharing later:

.. code-block:: python

   pip freeze > requirements.txt

This creates a list of installed packages.

--------------------------------------------------

You're Done!
------------

You’ve now:

- Installed Python and Git
- Set up a virtual environment
- Installed packages
- Ran the CTN-0094 Pipeline

If you had trouble, recheck the steps above. Most errors are caused by skipping small things (like forgetting to activate the virtual environment).

Welcome to the project!

Understanding the Pipeline Help Commands
=========================================

When you run the following command:

.. code-block:: python

   python run_pipelineV2.py --help

You’ll see a list of available options. This section breaks down each one and explains exactly what it does.

--------------------------------------------------

-d or --directory
------------------

**Required.** This is the folder where all results (predictions, logs, evaluations) will be saved.

.. code-block:: python

   -d ./results

If you don't create this folder manually, the script will make it for you. Think of this as the “output folder.”

What goes in here:
- A `logs/` folder with run details
- A `predictions/` folder with model output
- An `evaluations/` folder with accuracy, AUC, etc.

--------------------------------------------------

-s or --seed
-------------

This sets the **random seed** — a number used to make sure results are reproducible.

.. code-block:: python

   -s 42

Use any integer. This affects random parts of the pipeline like data splitting and sampling.

--------------------------------------------------

-l or --loop
-------------

Use this to run the pipeline multiple times — once per seed.

.. code-block:: python

   -l 1 5

This will run the pipeline 5 times, with seeds 1, 2, 3, 4, and 5.

**Why use it?** To see how your model performs under slightly different random sampling conditions.

--------------------------------------------------

-o or --outcomes
-----------------

List one or more outcomes (target variables) to model.

.. code-block:: python

   -o opioid_use_past_30days meth_use_past_30days

**What is an outcome?**  
An outcome is something you want to predict — like whether someone used opioids in the last 30 days, or their treatment completion status.

These outcomes are measured in the **CTN-0094 dataset** — a study of participants with opioid use disorder.

If you don’t use `-o`, the pipeline will run *all default outcomes*.

--------------------------------------------------

-m or --model
--------------

Let’s you choose the model to run. Options include:

.. code-block:: bash

   -m logistic_rf

This will run both **logistic regression** and **random forest**.

Supported models (more may be added over time):
- `logistic`
- `rf` (random forest)
- `nb` (negative binomial)
- `xgboost`

You can mix and match.

--------------------------------------------------

-dm or --demographic
---------------------

Set the **demographic group** to compare — for fairness evaluation.

.. code-block:: python

   -dm Race

This tells the pipeline to compare different racial/ethnic groups, like “non-Hispanic White” vs “Minority.”

**How it works:**  
The data is split into these two groups, and the pipeline evaluates whether the model performs equally well on both.

Other valid options (depending on your dataset):
- `Gender`
- `AgeGroup`
- `Ethnicity`

--------------------------------------------------

--quiet
--------

If you add `--quiet`, the pipeline will run with **minimal printouts**.

.. code-block:: python

   python run_pipelineV2.py ... --quiet

This is useful if you want a cleaner terminal while the script runs.

--------------------------------------------------

--progress
-----------

This adds a **progress bar** during the loop mode.

.. code-block:: python

   python run_pipelineV2.py -l 1 5 --progress

--------------------------------------------------

What Happens Behind the Scenes
===============================

Here’s what the script is actually doing:

1. **Loads the master dataset** — a clean, processed table of participants from the CTN-0094 study.

2. **Subsamples the data** — usually 1000 participants (with matching demographics).

3. **Joins the selected outcome** — such as `opioid_use_past_30days`.

4. **Preprocesses features** — including encoding, imputation, and scaling.

5. **Trains the model(s)** — one or more ML models depending on the `-m` flag.

6. **Evaluates performance** — using:
   - Accuracy
   - AUC (Area Under the Curve)
   - F1 score
   - Recall
   - Precision
   - *Fairness metrics* (how performance differs across demographic groups)

7. **Logs all results** — including seed, outcome, models, and metrics.

8. **Outputs** predictions and evaluation summaries.

--------------------------------------------------

Real Example
------------

Here’s a full example you can copy:

.. code-block:: python

   python run_pipelineV2.py \
       -d ./results \
       -l 1 3 \
       -o opioid_use_past_30days \
       -m logistic rf \
       -dm Race \
       --progress

What it does:
- Runs 3 loops (seeds 1, 2, 3)
- Uses logistic regression and random forest
- Predicts opioid use
- Evaluates fairness by Race
- Saves everything to `results/`

--------------------------------------------------

Need Help?
----------

If you’re not sure which outcomes to use, check the ``master_outcome_selections.csv`` file in the repo.

Or run the script once without any arguments to see the help screen again.

--------------------------------------------------

``Happy modeling!``

``CTN-0094 Team``
