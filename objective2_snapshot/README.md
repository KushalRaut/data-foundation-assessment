# HIT140 Objective 2

Completed Python analysis and extended Word report for the two required multiple linear regressions.

## Start here

- `output/Objective_2_Extended_Report.docx`: full report, with model definitions, variable selection, fitted equations, uncertainty, validation, diagnostics and limitations.
- `data/regression_2_1.csv`: 104 actual FIFA World Cup 2026 matches; ID + exactly eight predictors + goal difference.
- `data/regression_2_2.csv`: 208 team–match rows; ID + exactly eight predictors + goals.
- `data/data_dictionary.csv`: variable definitions and units.
- `data/matches_fifa.csv`: authoritative scores, dates, venues and all 104 source URLs.

## Reproduce

Use Python 3.12 or a compatible Python version. Install the recorded dependencies in a virtual environment:

```sh
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python build_data.py
python analyse.py
python validate.py
python create_report.py
```

All commands resolve project files relative to the script location. Raw data are included, so the analysis can run offline after dependencies are installed. The provided Word report has also been rendered and visually checked; document-rendering software is not needed to run the models.

## Sources

Primary tournament source: FIFA full-time match reports, all 104 PDFs in `raw/fifa/`. Resource IDs run from 12449 to 12552, but are **not** chronological match numbers. Read the match number inside each report. `collect_fifa.py` downloads these reports and extracts text, and can be used to refresh the archive. Refreshing can change hashes if FIFA corrects a file.

Historical source: Mart Jürisoo's international results, downloaded from
https://raw.githubusercontent.com/martj42/international_results/master/results.csv
on 5 October 2026. `raw/history_README.md` preserves source definitions. Historical scores include extra time and exclude shootouts. `build_data.py` retains only pre-11-June-2026 history, then appends current-tournament history from FIFA as matches occur. Two ambiguous historical rows are excluded and recorded. The external archive retains later records for provenance, but those never enter the feature builder.

A secondary fixtures webpage and an external Elo file were explored during research but **neither enters this analysis**. Elo is calculated in Python from archived prior results, using the documented fixed algorithm.

## Evaluation design

- Features are strictly earlier than each match's calendar date. No current-match statistics or final tournament totals are predictors.
- Training ends 2 July 2026: 85 matches / 170 team rows.
- Holdout starts 3 July: 19 matches / 38 team rows.
- Model coefficients remain frozen for the holdout. Earlier held-out outcomes may update features for later matches, as in sequential forecasting.
- Three chronological development folds keep entire dates and matches together.
- Exactly four shared concepts: Elo difference, rest difference, home advantage and competitive share. The difference and level versions of competitive share are counted as the same concept.
- Regression 2.1 coefficients use HC3 uncertainty; Regression 2.2 uses match-clustered uncertainty. Repeated-team dependence remains a limitation.
- Match bootstrap confidence intervals use seed 1402026 and 3,000 replicates, conditional on the trained model. They are not individual prediction intervals.
- Eight-predictor models are retained as required. Simpler models are explicitly labelled baselines, not replacements.

## Actual held-out results

| Model | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Goal difference | 1.146 | 1.395 | 0.128 |
| Team goals | 0.971 | 1.335 | 0.024 |

These are modest results. The team-goals model underperforms the Elo-only baseline. The report discusses that finding and does not assert high predictive accuracy or promise a grade.

## Evidence and checks

`feature_audit.csv` contains the prior ten matches used for every feature row's team histories. `source_comparison.csv` records agreement of FIFA outcomes with the historical source. `source_hashes.json` fingerprints used raw snapshots. `validate.py` checks dimensions, features, match pairing, score integrity, strict prior dates, split isolation and independently recalculated MAE/RMSE.

All coefficients, errors, correlations, descriptive statistics, influential observations, validation folds, charts, prediction tables and JSON summaries are in `output/`. The primary model keeps genuine outliers. Removing a single influential training match and clipping negative predictions are labelled sensitivity checks, not tuned replacements.

## Scope of this package

This is the requested extended Objective 2 analysis package. It does not contain team identities, a published repository URL, a signed AI declaration or evidence of group contributions. Those administrative items remain with the group as requested. The report exceeds the original four-page limit at the user's instruction. This ZIP is a project bundle, not a claim that the original submission packaging requirements have been fulfilled.
