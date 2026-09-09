# Defensive Effectiveness and Group-Stage Progression

## Analytic question

**Did teams that advanced from the FIFA World Cup 2026 group stage allow fewer shots on target per match than teams that were eliminated?**

This investigation uses shots on target conceded as a practical measure of defensive effectiveness. It is more precise to call this *defensive effectiveness* than *defensive pressure*: pressure itself would require variables such as pressures applied, turnovers forced, or PPDA.

## Data and sources

The analysis covers the complete 2026 World Cup group stage: 72 matches, 144 team-match records, and 48 teams. Each team played three group matches. Shots on target come from FIFA's team statistics. Progression status comes from the 32 teams listed in FIFA's Round of 32 match report hub.

- [FIFA group-stage report hub](https://www.fifatrainingcentre.com/en/fifa-world-cup-2026/match-report-hub.php)
- [FIFA knockout-stage report hub](https://www.fifatrainingcentre.com/en/fifa-world-cup-2026/match-report-hub-knockout-stage.php)
- [Example FIFA full-time report](https://fdp.fifa.org/assetspublic/ce281/r12452/pdf/FullTimeMatchReport-English.pdf)
- [FIFA team statistics](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics/team-statistics)

For 65 matches, the value was extracted from the team-statistics row of the FIFA full-time report: `Attempts at Goal (Total/On Target)`. Five inaccessible full-time reports were completed with FIFA Training Centre post-match summary reports. Scotland-Morocco and Cabo Verde-Saudi Arabia were completed from their FIFA official match pages. Every row retains its source URL.

The FIFA reports sometimes show small differences between the team-statistics row and the sum of player rows. The dataset consistently uses the team-statistics figure because it is the official match-level measure. This choice is recorded in the source files rather than silently mixing definitions.

## Variables and preparation

The raw dataset contains one row per match. Python reshapes each match into two team appearances. For a given team:

`shots on target conceded = opponent's shots on target`

The three appearances are then aggregated into one team-level observation:

`average shots on target conceded = total conceded across three group matches / 3`

The independent variable is progression status (`Advanced` or `Eliminated`). The dependent variable is average shots on target conceded per group-stage match. All 48 teams are retained; 0-0 matches remain relevant because both teams still attempted shots.

This is a census of teams in this tournament rather than a random sample. The inferential analysis can be interpreted as evidence about a broader conceptual population of comparable elite international tournament performances, not uncertainty about the recorded 48 teams themselves.

## Hypotheses and method

- **Null hypothesis (H0):** advancing and eliminated teams have equal population mean shots on target conceded per match.
- **Alternative hypothesis (H1):** their population means differ.
- **Test:** two-sided Welch independent-samples t-test, alpha = 0.05.
- **Interval:** 95% Welch confidence interval for `advanced mean - eliminated mean`.

Welch's version is used because group sizes differ (32 and 16) and equal population variances are not assumed.

## Results

| Progression | Teams | Mean | SD | Median | Minimum | Q1 | Q3 | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Advanced | 32 | 3.40 | 1.19 | 3.67 | 0.67 | 2.33 | 4.33 | 5.33 |
| Eliminated | 16 | 5.67 | 1.79 | 5.67 | 2.67 | 4.25 | 7.00 | 9.67 |

The estimated mean difference was **-2.27 shots on target per match**. Its 95% confidence interval was **[-3.30, -1.24]**. The Welch test produced **t(21.82) = -4.59, p = 0.000147**. Cohen's d was **-1.60**, indicating a large observed standardized difference.

At the 5% significance level, reject H0. The data provide strong evidence that the two population means differ. In this tournament, advancing teams allowed about 2.27 fewer shots on target per match on average. The direction of the estimate supports the proposed idea that better defensive effectiveness was associated with progression.

## Interpretation and limitations

The result shows association, not causation. Conceding fewer shots may help progression, but strong teams may also dominate possession, score first, or face different-quality opponents. Shots on target do not measure chance quality; a weak attempt and a clear one-on-one chance each count once.

The independence assumption is approximate. Teams played one another, so an attacking value for one team directly becomes a conceded value for its opponent. Each team contributes only one aggregated observation, which avoids treating its three matches as independent, but it cannot remove the tournament's network structure. With 32 of 48 teams advancing, the format also creates unequal group sizes. Results describe this tournament and should not automatically be generalized to other competitions.

## How to reproduce

From the repository root, after installing `requirements.txt`:

```bash
python topics/01_defensive_effectiveness/analysis.py
```

This single script rebuilds the match dataset from saved evidence, prepares the team records, computes descriptive statistics, a confidence interval and Welch's t-test, and saves the chart. Paths resolve relative to the script, so it can also run from another working directory.

## File guide

- `analysis.py`: complete dataset-building and analysis workflow.
- `data/`: source evidence, qualification records, and the rebuilt match-level CSV; all datasets are committed for reproducibility.
- `outputs/`: prepared team datasets, descriptive statistics, inferential results, and the chart.

## Assessment item still to resolve

This implementation uses a census of all 48 teams, not a probability sample. The brief explicitly requires data preparation and sampling. Before submission, agree on and document a defensible sampling strategy, its target population, sample size and reproducible random seed, then update the analysis and interpretation accordingly. The existing results below reflect the census approach. Source claims are retained from the original project and have not been independently reverified during repository reorganization.

## Suggested presentation script (about 2.5 minutes)

My question was whether teams that advanced from the 2026 World Cup group stage allowed fewer shots on target per match than eliminated teams. I used shots on target conceded as a measure of defensive effectiveness because it captures how often opponents produced attempts that required a save or resulted in a goal.

I collected the official FIFA statistics for all 72 group-stage matches. In Python, I converted each match into two team records. A team's shots on target conceded equal its opponent's shots on target. I then averaged each team's values across its three matches, giving one observation for each of the 48 teams. FIFA's Round of 32 listings classified 32 teams as advanced and 16 as eliminated. I checked for missing values, duplicate matches, team counts, and whether every team had exactly three appearances.

Advancing teams conceded an average of 3.40 shots on target per match, with a standard deviation of 1.19. Eliminated teams averaged 5.67, with a standard deviation of 1.79. The observed difference, advanced minus eliminated, was negative 2.27 shots per match.

I used a two-sided Welch t-test because the two groups had different sizes and I did not want to assume equal variances. The 95% confidence interval for the mean difference was negative 3.30 to negative 1.24. The test result was t equals negative 4.59 with about 21.82 degrees of freedom, and p equals 0.000147. Because the p-value is below 0.05 and the interval excludes zero, I rejected the null hypothesis. Advancing teams conceded significantly fewer shots on target on average.

This is an association and does not prove that limiting shots caused progression. Opponent quality, possession, scoring first, and goalkeeping may also matter. Shots on target also ignore chance quality. Still, the result supports the conclusion that defensive effectiveness clearly distinguished advancing teams in this tournament.

## AI use acknowledgement

Generative AI assisted with source discovery, data extraction, Python coding, statistical checking, documentation, and wording. Team members should verify the source records, run and understand the code, adapt the presentation in their own words, and describe this assistance accurately in the signed AI Usage Declaration Form.
