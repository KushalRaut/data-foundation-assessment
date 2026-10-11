# HIT140 — Objective 2: predicting World Cup match outcomes

This repository contains the datasets and Python code for **two multiple linear regression models**. It replaces the four Objective 1 analyses on `main`.

The previous work and the extended report are preserved on the [backup branch](https://github.com/KushalRaut/data-foundation-assessment/tree/backup/before-objective2-cleanup-2026-10-11). The original local folder is also unchanged. This repository does not include the separate Word submission or signed AI declaration.

## What are we trying to do?

For an illustrative Japan 3–1 South Korea match:

| | Regression 2.1 | Regression 2.2 |
|---|---|---|
| Question | What will the goal difference be? | How many goals will one team score? |
| Actual outcome in this example | Japan minus South Korea = +2 | Japan = 3; South Korea = 1 |
| Rows for this match | One | Two, one from each team's perspective |
| Complete dataset | 104 matches | 208 team–match observations |
| Predictors | Exactly eight | Exactly eight |

A predicted difference of +2 does not specify the score: 2–0 and 3–1 both have that difference. Model 2.2 is one fitted model applied separately to each team. Linear regression can predict decimal goals; evaluate these unrounded predictions. The two independently fitted models need not agree with each other.

## Run everything

Use Python 3.12 or later in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
# On Windows use: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_analysis.py
```

This runs three steps:

1. `build_data.py` reads the archived source records and rebuilds the two model datasets using only prior information.
2. `analyse.py` fits both models, evaluates predictions and saves statistics and charts.
3. `validate.py` checks dataset sizes, predictor counts, outcomes, historical dates, train/test separation and calculated errors.

All results go into `output/`, which Git ignores. No manual folder creation or internet connection is required after installing dependencies. Commands resolve files relative to their script location.

## Where should I look?

| File or folder | Purpose |
|---|---|
| `data/regression_2_1.csv` | 104 rows: match ID, eight predictors, actual goal difference |
| `data/regression_2_2.csv` | 208 rows: row ID, eight predictors, actual team goals |
| `data/matches_fifa.csv` | Match dates, teams, scores, venues and individual FIFA source URLs |
| `data/team_match_metadata.csv` | Connects each model 2.2 row to its match, team and opponent |
| `data/data_dictionary.csv` | Exact predictor definitions and units |
| `data/source_hashes.json` | Fingerprints of the archived source files, checked during rebuilding |
| `raw/history.csv` | Historical international match results used to calculate prior form and strength |
| `raw/history_README.md` | Historical source documentation |
| `raw/fifa/` | Original FIFA PDFs, extracted text used by the builder and URL index |
| `collect_fifa.py` | Optional source downloader; not required for normal reproduction |

The PDFs are source evidence, not generated analysis outputs. Keeping the archived inputs allows the calculations to be checked even if a website changes. IDs, dates and team names are metadata, not extra explanatory variables.

## The eight predictors for each model

All recent-form statistics refer to the last **10 internationals before the match date**.

| Model 2.1 predictor | Meaning |
|---|---|
| `elo_difference` | Team A's calculated strength rating minus Team B's, divided by 100 |
| `win_rate_difference` | Difference in recent proportions of matches won |
| `draw_rate_difference` | Difference in recent proportions drawn |
| `rest_difference` | Difference in days since the last match, with each team's rest capped at 14 days |
| `home_advantage` | +1 if A plays in its own country, −1 if B does, otherwise 0 |
| `world_cup_experience_difference` | Difference in log(1 + previous World Cup finals appearances at match level) |
| `head_to_head_balance` | A's result balance over up to five earlier meetings, smoothed toward zero |
| `competitive_share_difference` | Difference in recent fractions of matches classified as non-friendly |

| Model 2.2 predictor | Meaning |
|---|---|
| `elo_difference` | Team's calculated strength relative to its opponent, divided by 100 |
| `rest_difference` | Team's capped rest days minus opponent's capped rest days |
| `home_advantage` | +1 for playing in own country, −1 if the opponent does, otherwise 0 |
| `competitive_share` | Team's recent fraction of non-friendly matches |
| `goals_for_mean` | Team's recent average goals scored |
| `opponent_goals_against_mean` | Opponent's recent average goals conceded |
| `failed_to_score_rate` | Team's recent fraction of matches with zero goals scored |
| `opponent_clean_sheet_rate` | Opponent's recent fraction of matches with zero goals conceded |

There are **four shared concepts**: strength, rest, home advantage and competitive share. Competitive share counts as shared even though one model uses a difference and the other uses a level. These are modelling choices, not variables prescribed by the assignment.

Goals include extra time when played but exclude penalty-shootout kicks. First-listed team does not automatically mean home advantage. Past scoring averages are permitted because they precede the match; current-match scoring statistics cannot be predictors.

## How the data are built

Tournament outcomes come from the 104 archived [FIFA full-time reports](https://fdp.fifa.org/assetspublic/ce281/r12552/pdf/FullTimeMatchReport-English.pdf). Each match's exact link is in `matches_fifa.csv`. PDF resource IDs are not match numbers; the code reads the match number inside the report.

Prior results come from [Mart Jürisoo's international results dataset](https://github.com/martj42/international_results), using the archived CSV retrieved on 5 October 2026. Historical results before 11 June 2026 initialise team histories. Earlier 2026 World Cup results then update those histories as the competition progresses. Later rows in the archive are used only for a separate tournament score cross-check, never to initialise predictors. Two ambiguous duplicate historical records are excluded and listed in the generated audit.

The strength rating is an analyst-calculated Elo-style rating, not an official FIFA ranking. Every team starts at 1500; the update factor is 20 and the home adjustment is 100 rating points. Expected result is `1 / (1 + 10 ** ((opponent_rating - team_rating - home_adjustment) / 400))`. The rating changes by `20 × (actual_result − expected_result)`, where actual result is 1, 0.5 or 0. Shootouts do not turn drawn match scores into wins. All matches on a date use ratings and histories from before that date; updates happen afterwards.

World Cup experience is the number of earlier finals matches, including earlier 2026 matches, transformed with `log(1 + count)`. Head-to-head balance sums wins (+1), draws (0) and losses (−1) across up to five prior meetings, then divides by the number of those meetings plus two. With no prior meeting it is zero.

The raw files' recorded hashes are checked when rebuilding. The optional downloader should not be run merely to reproduce the analysis: a changed source must be investigated before deliberately updating its manifest.

## How we evaluate the models

Each model fits an intercept plus its eight predictors using ordinary least squares. It learns coefficients from **85 matches through 2 July** (170 rows for model 2.2), then predicts **19 later matches** (38 team rows). Both team rows of a match stay in the same split. Three earlier expanding time splits provide development checks.

The coefficients stay fixed during the final test period. Features for later test matches may include already completed test matches, because those outcomes would be known before the next match. This is sequential pre-match prediction, not a forecast of the entire knockout bracket made on one date.

Read these generated files first:

- `output/metrics_2_1.csv` and `metrics_2_2.csv`: prediction accuracy versus simple baselines.
- `output/predictions_2_1.csv` and `predictions_2_2.csv`: actual and predicted outcomes side by side.
- `output/coefficients_2_1.csv` and `coefficients_2_2.csv`: estimated effects of the predictors.
- `output/diagnostics_2_1.png` and `diagnostics_2_2.png`: residual and prediction plots.
- `output/feature_audit.csv`: the earlier match histories behind each row.

**MAE** is the average absolute prediction error, measured in goals. **RMSE** gives larger errors more weight. Lower values are better for both. Test **R²** compares errors with variation around the test-set mean; it can be negative. A separate training-mean baseline is also reported.

The additional outputs support the report: descriptive statistics, predictor correlations, VIF checks for overlapping predictor information, residual diagnostics, coefficient uncertainty, bootstrap uncertainty and sensitivity to an influential training match. Model 2.1 uses HC3 standard errors; model 2.2 groups standard errors by match because its two rows are related. These extensions do not eliminate dependence caused by teams appearing repeatedly.

The main results retain genuine high-scoring matches. Sensitivity analyses are labelled separately. Do not choose a model based on which one looks best on the final test set.

## How to explain the report

1. **Objective:** predict goal difference and team goals using pre-match information.
2. **Data:** explain the two row structures, sources, eight variables and prevention of future-information leakage.
3. **Method:** explain how regression estimates coefficients and why later matches are held out.
4. **Results:** show errors, baseline comparisons, coefficient interpretations and relevant charts.
5. **Limitations:** a small all-knockout test set, repeated teams, correlated predictors, historical data quality and differing match lengths when extra time occurs. Linear regression can also produce negative goal estimates.
6. **Conclusion:** say what the evidence supports; modest prediction accuracy is a result to explain, not something to hide.

The archived report is an extended draft. Check its numerical claims against freshly generated outputs before submission. Report formatting, the signed AI declaration and genuine collaboration records in your lecturer's Microsoft Teams space remain separate submission responsibilities.
