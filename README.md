# HIT140 Objective 2

Two multiple linear regression models predict FIFA World Cup 2026 outcomes using information available before each match. Each model uses exactly eight predictors, with no more than four shared predictor concepts.

| Model | Outcome | Dataset | Run online |
|---|---|---:|---|
| Regression 2.1 | Team A goals minus Team B goals | 104 match rows | [Open in Colab](https://colab.research.google.com/github/KushalRaut/data-foundation-assessment/blob/main/Regression_2.1/analysis.ipynb) |
| Regression 2.2 | Goals scored by one team | 208 team–match rows | [Open in Colab](https://colab.research.google.com/github/KushalRaut/data-foundation-assessment/blob/main/Regression_2.2/analysis.ipynb) |

## How to run

Open a Colab link, sign in if prompted, and select **Runtime → Run all**. Setup and data loading are automatic. Results and charts appear below the code. No paid GPU is needed.

You can also inspect the saved outputs directly in the [2.1 notebook](Regression_2.1/analysis.ipynb) or [2.2 notebook](Regression_2.2/analysis.ipynb) on GitHub.

## Folders

- **Data:** shared datasets, [variable definitions](Data/data_dictionary.csv), source records, and Python preparation and validation code.
- **Regression_2.1:** goal-difference analysis.
- **Regression_2.2:** team-goals analysis.

Each regression folder contains one analysis notebook (`analysis.ipynb`) and its dependency list (`requirements.txt`). Generated files go into that regression's `outputs/` folder.

## Data and method

Sources are [FIFA match reports](Data/matches_fifa.csv) and [historical international results](https://github.com/martj42/international_results). Archived inputs are in `Data/raw/`. Scores include extra time and exclude penalty shootouts.

Both models use 82 earlier matches for training and 22 later matches for testing. The notebooks explain predictor choices, exploration, chronological cross-validation, regression diagnostics and comparisons with other methods. The separate group report summarises the findings and limitations.

## Rebuild the datasets — optional

The notebooks use the included datasets. To rebuild and check them locally, install the dependencies in a virtual environment, then run:

```bash
python -m pip install -r Regression_2.1/requirements.txt
python Data/prepare_data.py
python Data/validate_data.py
```

The Git history preserves Vishan's original Regression 2.1 contribution and Madhav's Regression 2.2 contribution. AI assisted final integration, testing and report preparation; the group must review the work and declare that assistance.
