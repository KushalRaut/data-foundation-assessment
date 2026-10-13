# HIT140 Objective 2 — World Cup regression

Two multiple linear regression models use information available **before a match**:

- **Regression 2.1:** predict Team A goals minus Team B goals, using 104 match rows.
- **Regression 2.2:** predict one team's goals, using 208 team–match rows.

Each required model has exactly eight predictors. Three predictor columns are identical across models; competitive-match share is a fourth shared underlying concept, counted conservatively despite the difference-versus-level forms.

## Run online — no local installation

1. Open a notebook below in Google Colab.
2. Sign in to Google if prompted, then choose **Runtime → Run all**. A standard CPU runtime is sufficient.
3. Read the tables and charts below each section. The notebook downloads the public repository and installs its packages automatically; do not upload data manually.

| Model | Run and edit online | Inspect saved code and outputs |
|---|---|---|
| Goal difference | [Open Regression 2.1 in Colab](https://colab.research.google.com/github/KushalRaut/data-foundation-assessment/blob/main/Regression_2.1/analysis.ipynb) | [GitHub notebook](Regression_2.1/analysis.ipynb) |
| Team goals | [Open Regression 2.2 in Colab](https://colab.research.google.com/github/KushalRaut/data-foundation-assessment/blob/main/Regression_2.2/analysis.ipynb) | [GitHub notebook](Regression_2.2/analysis.ipynb) |

GitHub displays a saved example run. Colab runs a new copy in a temporary cloud environment; use **Save a copy in Drive** to retain your own changes. Colab setup keeps compatible preinstalled packages to avoid replacing libraries already loaded in memory, and prints the versions used. Exact locally tested versions are recorded in each requirements file; minor numerical differences between versions are possible. Runtime availability and sign-in are controlled by Google. See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Repository layout

```text
Data/
  regression_2_1.csv             # ID + 8 predictors + goal difference
  regression_2_2.csv             # ID + 8 predictors + team goals
  matches_fifa.csv               # Dates, teams, actual scores and source links
  team_match_metadata.csv        # Teams and matches behind the model 2.2 IDs
  data_dictionary.csv            # Exact predictor definitions and units
  prepare_data.py                # Rebuild features from earlier matches
  validate_data.py               # Source, leakage and outcome checks
  source_hashes.json             # Integrity fingerprints for raw source records
  raw/                          # Archived FIFA reports and international results
Regression_2.1/
  analysis.ipynb                 # Explained notebook with saved example results
  analysis.py                    # Equivalent standalone Python workflow
  requirements.txt
Regression_2.2/
  analysis.ipynb
  analysis.py
  requirements.txt
README.md
```

Each regression creates its own `outputs/` folder when run. Separate generated charts and tables are ignored by Git; saved notebook outputs are included for immediate inspection. All data are shared in `Data/`, so there are no conflicting copies inside the regression folders. The separate Word report is supplied with the assessment submission.

## What the variables mean

All recent rates and averages use the latest **10 internationals strictly before the match date**. Proportions range from 0 to 1. Team A is the first-listed side in the FIFA report, not necessarily the home team. Scores include extra time but exclude penalty-shootout kicks.

| Regression 2.1 | Why it might predict goal difference |
|---|---|
| Elo difference / 100 | Relative long-term strength |
| Recent win-rate difference | Relative recent success |
| Recent draw-rate difference | Relative tendency to draw |
| Capped rest-day difference | Relative recovery time |
| Home advantage (+1, 0, −1) | Playing in the team's own country |
| Log World Cup experience difference | Relative experience in finals matches |
| Smoothed head-to-head balance | Previous results between the opponents |
| Competitive-match share difference | Relative exposure to non-friendly matches |

| Regression 2.2 | Why it might predict a team's goals |
|---|---|
| Elo difference / 100 | Strength relative to the opponent |
| Capped rest-day difference | Relative recovery time |
| Home advantage (+1, 0, −1) | Venue advantage |
| Team competitive-match share | Recent match context |
| Team average goals scored | Recent attacking output |
| Opponent average goals conceded | Recent defensive vulnerability |
| Team failure-to-score rate | Frequency of scoring no goals |
| Opponent clean-sheet rate | Frequency of preventing all goals |

The [data dictionary](Data/data_dictionary.csv) defines every calculation. Rest is capped at 14 days. Experience is `log(1 + prior finals matches)`. Head-to-head sums win/draw/loss signs over up to five prior meetings and divides by the number of those meetings plus two; no meetings gives zero.

## Sources and preparation

FIFA match reports supply the 104 outcomes, dates and venues; each exact URL appears in [matches_fifa.csv](Data/matches_fifa.csv). The archived PDFs and their extracted text are in `Data/raw/fifa/`. PDF resource IDs are not match numbers: the builder reads the number inside each report.

[Mart Jürisoo's international results](https://github.com/martj42/international_results) supplies the historical match sequence used to initialise form, experience and strength. The tournament match reports alone do not provide that full pre-tournament history. The snapshot, retrieved on 5 October 2026, is included with its source README. Only records before 11 June initialise predictors; earlier World Cup results are then added as the tournament progresses. Later historical-source rows are used only to cross-check tournament scores and dates. Two ambiguous historical records are excluded and recorded in the generated audit.

Elo is calculated by the team workflow, **not an official FIFA ranking**. Start at 1500, use K = 20 and a home adjustment of 100 rating points. Expected result is `1 / (1 + 10 ** ((opponent_rating - team_rating - home_adjustment) / 400))`; update by `20 × (actual_result − expected_result)`, where actual result is 1, 0.5 or 0. Every match on a given date uses earlier-date information; updates occur after all features for that date are extracted.

Rebuilding produces the exact committed model datasets. The audits check source hashes, all 104 PDF headers, both outcomes, paired team rows and the dates of all ten prior matches per team. No current-match statistics or final tournament aggregates are predictors.

## Evaluation and findings

The date nearest an 80% match split is **1 July 2026**: 82 training matches and 22 later test matches. Regression 2.2 therefore has 164 training and 44 test rows. Whole dates and both rows of a match stay together. The test sample consists entirely of knockout matches.

Exploration uses training data only: descriptive statistics, predictor–outcome scatterplots and Pearson correlations. The football-based eight-predictor sets are retained; weak correlations and predictor overlap are reported, not concealed. Three expanding validation folds use training matches only: 36→12, 48→18 and 66→16 training→validation matches. Scaling for ridge regression is fitted inside each fold.

OLS is the required method. Additional comparisons use ridge penalties 0.1, 1 and 10, and random forests with 200 trees, depth 3 or unlimited, and minimum leaf sizes 3 or 6 (seed 42). Select settings by validation MAE weighted by the number of validation rows. Compare with training-mean and Elo-only baselines on the same splits. Coefficients remain fixed during testing; features may include earlier test results that are known before a later match.

| Required OLS model | Test MAE | Test RMSE | Test R² | Training-mean MAE |
|---|---:|---:|---:|---:|
| Goal difference | 1.051 | 1.317 | 0.353 | 1.466 |
| Team goals | 0.926 | 1.272 | 0.075 | 1.065 |

MAE is average absolute error in goals; RMSE penalises large errors more. Lower is better for both. Test R² measures performance relative to variation around the test mean; it can be negative. Outcomes differ between the two tasks, so their MAEs are not a direct ranking of model quality.

Ridge selected penalty 10 and the forest selected depth 3 / leaf size 3 for both tasks. Elo-only had the best cross-validation MAE in both tasks; it remains a baseline because it does not meet the eight-predictor requirement. The forest achieved the lowest test MAE, but this alone does not justify selecting it for future use. These comparisons extend a project whose test outcomes were already examined; they are exploratory and require independent future validation.

Coefficient uncertainty uses HC3 standard errors for 2.1 and match-clustered standard errors for 2.2. Diagnostics include residual and Q–Q plots, VIF, Cook's distance and residual tests. A sensitivity fit omits the most influential training match without replacing the primary fit. Approximate 95% intervals use 3,000 whole-match bootstrap resamples (seed 42). Repeated-team dependence is not fully resolved. Other limitations are the small knockout-only test set, correlated predictors, historical data limitations, extra-time exposure and the possibility of negative goal predictions from linear regression.

## Run locally or rebuild the sources

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r Regression_2.1/requirements.txt
python Data/prepare_data.py
python Regression_2.1/analysis.py
python Regression_2.2/analysis.py
python Data/validate_data.py
```

Source rebuilding is optional for running the notebooks because the prepared datasets are included. `Data/audit/` is generated locally and ignored. Source snapshots must not be silently replaced: investigate hash mismatches first.

## Collaboration and integration

The merged Git history retains the team's submissions:

- Vishan Budhathoki's Regression 2.1 notebook: `cd8f626`.
- Madhav Paudel's Regression 2.2 script and statistical extensions: `06d01cf` and `fa896f8`.

Final integration organised the folders, adapted Regression 2.1 to the same date-based split, added training-only EDA and competing methods, checked data provenance, and provided executable notebooks. Both final workflows follow the teammates' load/check/split/fit/evaluate/interpret structure; repetitive plotting and validation were reorganised. AI assisted the integration, checks and report preparation. Each member should review and explain their actual contribution and record collaboration in the lecturer-provided Microsoft Teams space. Repository authorship is not a substitute for that evidence.

## Submission

Submit the four-page A4, single-spaced, 10 pt Word group report; a text file with this repository URL; the actual Turnitin originality report; and the AI declaration signed by all members in one ZIP. Add full names and IDs to the report before submission. No code screenshots or separate reference list are needed in the report; concise source attribution is still included.

Method documentation: [scikit-learn random forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), [chronological splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), and [statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html).
