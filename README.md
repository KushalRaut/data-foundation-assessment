# data-foundation-assessment

Group assessment: four distinct Python analyses of FIFA World Cup 2026 data.

## Repository layout

```text
topics/
  01_defensive_effectiveness/
    README.md       # Question, methods, results and limitations
    analysis.py     # Entire topic workflow
    data/           # Source evidence and input datasets
    outputs/        # Prepared datasets, statistical results and charts
requirements.txt
```

Use one topic folder per team member and one Python script per topic. Keep source data in `data/`, code beside that folder, and generated results in `outputs/`. The other three team members will upload their own topic folders. Each topic must investigate a different focal variable; changing groups while analysing the same variable is not sufficient.

## Run the existing topic

Python 3.10 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python topics/01_defensive_effectiveness/analysis.py
```

The defensive-effectiveness workflow rebuilds its dataset from committed evidence and runs without downloading data. Outputs are committed so reviewers can inspect the datasets and results. The defensive-effectiveness topic belongs to **Kushal Raut (student number: 399210)**. The other three topics will be uploaded by the respective team members.

## Required for every topic

1. A distinct analytic question.
2. Python data wrangling with documented sources and validation.
3. Preparation and sampling: population, sample, sampling method and justification.
4. Descriptive statistics.
5. A confidence interval with an interpretation.
6. A justified one-sample or two-sample t-test, hypotheses, significance level and conclusion.

The existing defensive topic uses all 48 teams as a census. Its sampling requirement still needs to be resolved before submission; see its topic README. This repository organization does not constitute four completed analyses.

## Submission checklist

Due: **10 September 2026, 14:00 ACST (Darwin)**. Group size: four students.

- Complete and review all four topics; include all Python files and datasets in the repository.
- Record one unlisted YouTube group presentation, with individual segments of at most three minutes and total duration no more than 12 minutes, +/- 30 seconds. Each presenter starts by showing CDU student ID, shows their face throughout and uses PowerPoint slides.
- Include methodology, findings, lessons learned and an overall conclusion.
- Submit one ZIP through Learnline containing the YouTube link, a text file linking to this repository, and the AI Usage Declaration Form signed by all members.
- Acknowledge AI assistance accurately, include a reflective account and conversation URL where applicable, and ensure every member understands their code.
- Document contributions and the creative process in the lecturer-provided Microsoft Teams space, as required by the brief.

Repository: https://github.com/KushalRaut/data-foundation-assessment
