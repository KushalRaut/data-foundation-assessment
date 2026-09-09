"""Analyse shots on target conceded and 2026 World Cup group-stage progression."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


# Full-time reports that could not be extracted are completed with the official
# FIFA match page / FIFA Training Centre post-match summary report.
FALLBACK = {
    3: (4, 2, "FIFA Training Centre post-match summary"),
    8: (7, 3, "FIFA Training Centre post-match summary"),
    10: (4, 7, "FIFA Training Centre post-match summary"),
    14: (0, 2, "FIFA official match page"),
    25: (4, 1, "FIFA Training Centre post-match summary"),
    28: (2, 3, "FIFA Training Centre post-match summary"),
    46: (2, 3, "FIFA official match page"),
}


def clean_team(name: str) -> str:
    return name.replace("\xa0", " ").strip()


def build_dataset() -> None:
    records = json.loads((DATA / "source_records.json").read_text())
    rows = []

    for record in records:
        i = record["index"]
        excerpt = record["fulltime_excerpt"]
        match = re.search(
            r"(\d+)\s*/\s*(\d+) Attempts at Goal \(Total/On Target\) "
            r"(\d+)\s*/\s*(\d+)",
            excerpt,
        )
        if match:
            total_a, sot_a, total_b, sot_b = map(int, match.groups())
            evidence = "FIFA full-time match report, team statistics row"
            evidence_url = record["fulltime_url"]
        else:
            sot_a, sot_b, evidence = FALLBACK[i]
            total_a = total_b = pd.NA
            evidence_url = (
                record["match_url"]
                if "match page" in evidence
                else record["summary_url"]
            )

        rows.append(
            {
                "match_id": record["match_id"],
                "date": record["date"],
                "group": record["group"],
                "team_a": clean_team(record["team_a"]),
                "team_b": clean_team(record["team_b"]),
                "score": record["score"],
                "team_a_total_attempts": total_a,
                "team_a_shots_on_target": sot_a,
                "team_b_total_attempts": total_b,
                "team_b_shots_on_target": sot_b,
                "evidence_type": evidence,
                "evidence_url": evidence_url,
                "fifa_match_url": record["match_url"],
                "fifa_fulltime_report_url": record["fulltime_url"],
                "fifa_summary_report_url": record["summary_url"],
            }
        )

    output = pd.DataFrame(rows)
    assert len(output) == 72
    assert output[["team_a_shots_on_target", "team_b_shots_on_target"]].notna().all().all()
    assert output["group"].value_counts().eq(6).all()
    teams = set(output["team_a"]) | set(output["team_b"])
    assert len(teams) == 48
    output.to_csv(DATA / "match_level_shots_on_target.csv", index=False)
    print(f"Saved {len(output)} matches covering {len(teams)} teams.")


def descriptive(values: pd.Series) -> dict[str, float]:
    return {
        "n": int(values.count()),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "q3": float(values.quantile(0.75)),
        "maximum": float(values.max()),
    }


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    build_dataset()
    matches = pd.read_csv(DATA / "match_level_shots_on_target.csv")
    qualification = json.loads((DATA / "qualified_teams.json").read_text())
    advanced = {team for pair in qualification["pairs"] for team in pair}

    # Data wrangling: turn each match into one record per team. The opponent's
    # shots on target are the team's shots on target conceded.
    a = matches[["match_id", "group", "team_a", "team_b", "team_b_shots_on_target"]].copy()
    a.columns = ["match_id", "group", "team", "opponent", "shots_on_target_conceded"]
    b = matches[["match_id", "group", "team_b", "team_a", "team_a_shots_on_target"]].copy()
    b.columns = ["match_id", "group", "team", "opponent", "shots_on_target_conceded"]
    appearances = pd.concat([a, b], ignore_index=True)
    appearances["advanced"] = appearances["team"].isin(advanced)
    appearances["progression"] = np.where(appearances["advanced"], "Advanced", "Eliminated")

    assert len(appearances) == 144
    assert appearances.isna().sum().sum() == 0
    assert appearances.groupby("team").size().eq(3).all()
    assert appearances["team"].nunique() == 48
    assert len(advanced) == 32 and advanced.issubset(set(appearances["team"]))

    # Sampling unit: one team, using its mean over all three group matches.
    teams = (
        appearances.groupby(["group", "team", "advanced", "progression"], as_index=False)
        .agg(
            matches_played=("match_id", "nunique"),
            total_shots_on_target_conceded=("shots_on_target_conceded", "sum"),
            average_shots_on_target_conceded=("shots_on_target_conceded", "mean"),
        )
        .sort_values(["advanced", "average_shots_on_target_conceded", "team"], ascending=[False, True, True])
    )
    assert len(teams) == 48 and teams["matches_played"].eq(3).all()

    advanced_values = teams.loc[teams["advanced"], "average_shots_on_target_conceded"]
    eliminated_values = teams.loc[~teams["advanced"], "average_shots_on_target_conceded"]

    result = stats.ttest_ind(advanced_values, eliminated_values, equal_var=False)
    ci = result.confidence_interval(confidence_level=0.95)
    difference = float(advanced_values.mean() - eliminated_values.mean())
    effect_size = difference / np.sqrt(
        ((len(advanced_values) - 1) * advanced_values.var(ddof=1)
         + (len(eliminated_values) - 1) * eliminated_values.var(ddof=1))
        / (len(advanced_values) + len(eliminated_values) - 2)
    )

    desc = pd.DataFrame(
        [
            {"progression": "Advanced", **descriptive(advanced_values)},
            {"progression": "Eliminated", **descriptive(eliminated_values)},
        ]
    )
    inference = {
        "question": "Did advancing teams allow fewer shots on target per group-stage match than eliminated teams?",
        "unit_of_analysis": "Team (mean across all three group-stage matches)",
        "advanced_n": int(len(advanced_values)),
        "eliminated_n": int(len(eliminated_values)),
        "mean_difference_advanced_minus_eliminated": difference,
        "confidence_level": 0.95,
        "ci_low": float(ci.low),
        "ci_high": float(ci.high),
        "test": "Two-sided Welch independent-samples t-test",
        "null_hypothesis": "The population mean shots on target conceded per match is equal for advancing and eliminated teams.",
        "alternative_hypothesis": "The population means differ.",
        "t_statistic": float(result.statistic),
        "degrees_of_freedom": float(result.df),
        "p_value": float(result.pvalue),
        "alpha": 0.05,
        "cohens_d": float(effect_size),
        "decision": "Reject H0" if result.pvalue < 0.05 else "Fail to reject H0",
    }

    appearances.to_csv(OUTPUTS / "team_match_records.csv", index=False)
    teams.to_csv(OUTPUTS / "team_level_analysis.csv", index=False)
    desc.to_csv(OUTPUTS / "descriptive_statistics.csv", index=False)
    (OUTPUTS / "inferential_results.json").write_text(json.dumps(inference, indent=2))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    groups = [advanced_values, eliminated_values]
    labels = ["Advanced\n(n=32)", "Eliminated\n(n=16)"]
    box = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.48, showmeans=True,
                     meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#152238"})
    for patch, color in zip(box["boxes"], ["#1976D2", "#EF6C00"]):
        patch.set_facecolor(color); patch.set_alpha(0.78)
    rng = np.random.default_rng(2026)
    for x, values, color in zip([1, 2], groups, ["#0D47A1", "#BF360C"]):
        ax.scatter(rng.normal(x, 0.045, len(values)), values, s=25, alpha=0.68, color=color, zorder=3)
    ax.set_title("Shots on Target Conceded by Group-Stage Progression")
    ax.set_ylabel("Average shots on target conceded per match")
    ax.set_xlabel("Team progression status")
    p_label = "p < 0.001" if result.pvalue < 0.001 else f"p = {result.pvalue:.3f}"
    ax.text(0.99, 0.98, f"Welch test: {p_label}\nMean difference = {difference:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#B0BEC5"})
    fig.tight_layout()
    fig.savefig(OUTPUTS / "defensive_effectiveness_boxplot.png", dpi=220)
    plt.close(fig)

    print(desc.to_string(index=False))
    print(json.dumps(inference, indent=2))


if __name__ == "__main__":
    main()
