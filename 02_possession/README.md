# Possession and match outcome

**Contributor:** name and student number not supplied in the source files.

## Question

Does average possession differ between winning and losing teams in the supplied FIFA World Cup 2026 match data?

## Run

From the repository root:

```bash
python 02_possession/analyse_possession.py
```

`data/` contains the original matches, team statistics and team identifiers. The script joins these inputs, removes drawn matches, compares possession, and saves prepared CSVs, three charts and statistical results to `outputs/`. The original Downloads folder remains unchanged.

The workflow consolidates `create_dataset.py`, `matches_nodraw.py` and `calculation.py`. The redundant mean-only calculation is covered by the full statistical summary. Charts are saved without blocking on interactive windows. Original calculations and the author's AI acknowledgement are retained.

## Author review needed

- Record the original data source URLs and extraction dates; these were not provided in the source folder.
- Document the population and sampling strategy. Filtering out draws is not random sampling.
- Winner and loser observations are paired within each match, with teams also appearing repeatedly. The supplied independent two-sample t-test does not account for this dependence. Review the design before interpreting its p-value.
- The original mean confidence intervals use 1.96 times the standard error (a normal approximation), rather than a Student t critical value.

These issues are documented rather than silently changing the teammate's analysis during organization.
