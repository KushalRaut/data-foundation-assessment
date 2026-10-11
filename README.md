# HIT140 — Objective 2

This branch currently contains the prepared datasets only. Python analysis and the report are the next tasks.

## Datasets

| File | Contents |
|---|---|
| `data/regression_2_1.csv` | 104 match rows: match ID, eight predictors and goal difference (Team A minus Team B). |
| `data/regression_2_2.csv` | 208 team–match rows: row ID, eight predictors and goals scored by that team. |
| `data/matches_fifa.csv` | Match dates, teams, scores, venues and FIFA source URLs. |
| `data/team_match_metadata.csv` | Links each team row to its match, team and opponent. |
| `data/data_dictionary.csv` | Predictor definitions and units. |

Predictors use information from before each match. The two models share four predictor concepts: strength, rest, home advantage and competitive-match share. Goals include extra time but exclude penalty shootouts.

Sources: FIFA match reports (individual links in `matches_fifa.csv`) and [historical international results](https://github.com/martj42/international_results). Archived source files and the existing Python draft are preserved on the [working-code backup branch](https://github.com/KushalRaut/data-foundation-assessment/tree/backup/objective2-working-code-2026-10-11).

## Next steps

1. Review the sources, dataset columns and predictor definitions together.
2. Agree on individual responsibilities and record actual contributions in the lecturer's Microsoft Teams space.
3. Add Python code to prepare and validate the datasets, checking that all predictors use prior information.
4. Build regression 2.1 to predict goal difference and regression 2.2 to predict team goals, using exactly eight predictors each.
5. Evaluate on later matches, keeping both team rows of each match together; compare prediction errors with simple baselines.
6. Create charts, interpret the results and limitations, and write the group report.
7. Review each other's work, commit each member's actual contributions, and prepare the required submission files.
