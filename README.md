# data-foundation-assessment

Group assessment: four distinct Python analyses of FIFA World Cup 2026 data.

## Repository layout

```text
01_defensive_effectiveness_Kushal_399210/          # Kushal Raut — 399210 (.py)
02_possession/                       # Teammate name to be supplied (.py)
03_fouls_suffered/                   # Teammate name to be supplied (.ipynb)
04_goals_conceded_by_confederation_Kuber_399136/   # Kuber Budhathoki — s399136 (.ipynb)
requirements.txt
```

Each topic has one analysis file beside `data/`, `outputs/` and a README explaining how to run it. Source datasets are kept in `data/`; generated datasets, statistics and charts go in `outputs/`. Notebooks also retain inline results.

## Run the analyses

Python 3.10 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 01_defensive_effectiveness_Kushal_399210/analyse_defensive_effectiveness.py
python 02_possession/analyse_possession.py
jupyter lab
```

In Jupyter, open each topic's notebook and run all cells in order. The notebooks support launching from the repository root or their own topic folder.

A `.py` file is a regular Python script that runs the workflow from start to finish. An `.ipynb` file is a Jupyter notebook containing code cells, explanatory text and saved outputs. Both use Python; the supplied assignment brief requires access to Python code and datasets without specifying a notebook format. Existing teammate notebooks are preserved, and the defensive topic remains a single commented script.

## Required for every topic

1. A distinct analytic question.
2. Python data wrangling with documented sources and validation.
3. Preparation and sampling: population, sample, sampling method and justification.
4. Descriptive statistics.
5. A confidence interval with an interpretation.
6. A justified one-sample or two-sample t-test, hypotheses, significance level and conclusion.

The existing defensive topic uses all 48 teams as a census. Its sampling requirement still needs to be resolved before submission; see its topic README. All four contributions are now included. Review each topic README for outstanding methodological or source-documentation items before submission.

## Submission checklist

Due: **10 September 2026, 14:00 ACST (Darwin)**. Group size: four students.

- Complete and review all four topics; include all Python files and datasets in the repository.
- Record one unlisted YouTube group presentation, with individual segments of at most three minutes and total duration no more than 12 minutes, +/- 30 seconds. Each presenter starts by showing CDU student ID, shows their face throughout and uses PowerPoint slides.
- Include methodology, findings, lessons learned and an overall conclusion.
- Submit one ZIP through Learnline containing the YouTube link, a text file linking to this repository, and the AI Usage Declaration Form signed by all members.
- Acknowledge AI assistance accurately, include a reflective account and conversation URL where applicable, and ensure every member understands their code.
- Document contributions and the creative process in the lecturer-provided Microsoft Teams space, as required by the brief.

Repository: https://github.com/KushalRaut/data-foundation-assessment
