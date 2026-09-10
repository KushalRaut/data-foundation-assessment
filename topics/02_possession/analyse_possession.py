"""Compare possession in winning and losing match appearances.

Consolidated from the teammate's create_dataset.py, matches_nodraw.py and
calculation.py. Original statistical calculations are retained.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# ---------------------------------------------------------------
# Shared visual theme (matches the presentation's color palette)
# ---------------------------------------------------------------
NAVY = "#0B3D2E"     # winning teams
RED = "#B23A48"      # losing teams
GOLD = "#FFC72C"      # accent / error bars
GRID = "#D8DED9"
TEXT = "#1A1A1A"
MUTED = "#5B6B63"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def calculate_statistics(df, save_dir="."):
    """
    Compares possession (%) between winning and losing teams:
    mean, standard deviation, standard error, 95% CI, and an
    independent-samples t-test. Saves three chart images
    (histogram, bar+CI, box plot) ready to drop into the slides.
    """

    winning, losing = [], []

    for _, row in df.iterrows():
        if row["Goals A"] > row["Goals B"]:
            winning.append(row["Possession A"])
            losing.append(row["Possession B"])
        else:
            winning.append(row["Possession B"])
            losing.append(row["Possession A"])

    # ---------------- Mean ----------------
    winning_mean = sum(winning) / len(winning)
    losing_mean = sum(losing) / len(losing)

    # ---------------- Standard deviation ----------------
    winning_sd = pd.Series(winning).std()
    losing_sd = pd.Series(losing).std()

    # ---------------- Standard error & 95% CI ----------------
    n_winning, n_losing = len(winning), len(losing)
    winning_se = winning_sd / (n_winning ** 0.5)
    losing_se = losing_sd / (n_losing ** 0.5)

    winning_margin = 1.96 * winning_se
    losing_margin = 1.96 * losing_se

    winning_lower, winning_upper = winning_mean - winning_margin, winning_mean + winning_margin
    losing_lower, losing_upper = losing_mean - losing_margin, losing_mean + losing_margin

    # ---------------- Independent t-test ----------------
    t_stat, p_value = ttest_ind(winning, losing)

    # ---------------- Print results ----------------
    print("\n--- Results ---")
    print("Number of winning teams:", n_winning)
    print("Number of losing teams:", n_losing)
    print("\nAverage winning possession:", round(winning_mean, 2))
    print("Average losing possession:", round(losing_mean, 2))
    print("\nWinning standard deviation:", round(winning_sd, 2))
    print("Losing standard deviation:", round(losing_sd, 2))
    print("\nWinning 95% CI:", round(winning_lower, 2), "to", round(winning_upper, 2))
    print("Losing 95% CI:", round(losing_lower, 2), "to", round(losing_upper, 2))
    print("\nT-statistic:", round(t_stat, 3))
    print("P-value:", round(p_value, 4))

    # for the graph design and some syntax i have use Ai and i have done all the calculation for this project

    if p_value < 0.05:
        print("\nReject H0 — significant difference between winning and losing possession.")
    else:
        print("\nFail to reject H0 — no significant difference between winning and losing possession.")

    # GRAPH 1 — HISTOGRAM

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)

    bins = 10
    ax.hist(winning, bins=bins, alpha=0.75, label="Winning", color=NAVY, edgecolor="white")
    ax.hist(losing, bins=bins, alpha=0.75, label="Losing", color=RED, edgecolor="white")

    ax.axvline(winning_mean, color=NAVY, linestyle="--", linewidth=1.6)
    ax.axvline(losing_mean, color=RED, linestyle="--", linewidth=1.6)

    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.18)  # headroom so mean labels never collide with bars
    win_right = winning_mean >= losing_mean
    ax.text(winning_mean, top * 1.12, f" mean {winning_mean:.1f}%",
            color=NAVY, fontsize=9, fontweight="bold", va="top",
            ha="left" if win_right else "right")
    ax.text(losing_mean, top * 1.12, f" mean {losing_mean:.1f}%",
            color=RED, fontsize=9, fontweight="bold", va="top",
            ha="right" if win_right else "left")

    ax.set_xlabel("Possession (%)", fontsize=11)
    ax.set_ylabel("Number of teams", fontsize=11)
    ax.set_title("Distribution of Possession: Winning vs Losing Teams", fontsize=13, pad=14)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(0.0, 0.98))

    fig.tight_layout()
    fig.savefig(f"{save_dir}/chart_histogram.png", bbox_inches="tight")
    plt.close(fig)


    # GRAPH 2 — BAR CHART WITH 95% CI

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)

    means = [winning_mean, losing_mean]
    errors = [winning_margin, losing_margin]
    colors = [NAVY, RED]
    labels = ["Winning", "Losing"]

    bars = ax.bar(labels, means, yerr=errors, capsize=6, color=colors, width=0.55,
                   error_kw={"ecolor": GOLD, "elinewidth": 2, "capthick": 2})

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + max(errors) + 1.2,
                 f"{mean:.1f}%", ha="center", fontsize=11, fontweight="bold", color=TEXT)

    ax.set_ylabel("Mean Possession (%)", fontsize=11)
    ax.set_title("Mean Possession \u00b1 95% CI: Winning vs Losing Teams", fontsize=13, pad=14)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_ylim(0, max(means) + max(errors) + 8)

    fig.tight_layout()
    fig.savefig(f"{save_dir}/chart_bar_ci.png", bbox_inches="tight")
    plt.close(fig)

    # GRAPH 3 — BOX PLOT
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)

    bp = ax.boxplot([winning, losing], tick_labels=["Winning", "Losing"], patch_artist=True,
                     widths=0.5, medianprops={"color": GOLD, "linewidth": 2.2},
                     boxprops={"linewidth": 1.2}, whiskerprops={"color": MUTED, "linewidth": 1.2},
                     capprops={"color": MUTED, "linewidth": 1.2},
                     flierprops={"markerfacecolor": MUTED, "markeredgecolor": "none", "markersize": 5})

    for patch, color in zip(bp["boxes"], [NAVY, RED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_ylabel("Possession (%)", fontsize=11)
    ax.set_title("Possession Spread: Winning vs Losing Teams", fontsize=13, pad=14)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{save_dir}/chart_boxplot.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "n_winning": n_winning, "n_losing": n_losing,
        "winning_mean": winning_mean, "losing_mean": losing_mean,
        "winning_sd": winning_sd, "losing_sd": losing_sd,
        "winning_ci": (winning_lower, winning_upper),
        "losing_ci": (losing_lower, losing_upper),
        "t_stat": t_stat, "p_value": p_value,
    }





def build_dataset():
    """Join match scores, team identities and possession, then exclude draws."""
    # Load the three CSV files
    matches = pd.read_csv(DATA / "matches.csv")
    team_stats = pd.read_csv(DATA / "match_team_stats.csv")
    teams = pd.read_csv(DATA / "teams.csv")



    # 1. Add team names to match_team_stats


    team_stats = team_stats.merge(
        teams[["team_id", "team_name"]],
        on="team_id",
        how="left"
    )


    # 2. Get Team A statistics


    team_a = team_stats[
        ["match_id", "team_id", "team_name", "possession_pct"]
    ].copy()

    team_a = team_a.rename(columns={
        "team_id": "home_team_id",
        "team_name": "Team A",
        "possession_pct": "Possession A"
    })



    # 3. Get Team B statistics


    team_b = team_stats[
        ["match_id", "team_id", "team_name", "possession_pct"]
    ].copy()

    team_b = team_b.rename(columns={
        "team_id": "away_team_id",
        "team_name": "Team B",
        "possession_pct": "Possession B"
    })



    # 4. Merge Team A and Team B possession


    final = matches.merge(
        team_a[["match_id", "home_team_id", "Team A", "Possession A"]],
        on=["match_id", "home_team_id"],
        how="left"
    )

    final = final.merge(
        team_b[["match_id", "away_team_id", "Team B", "Possession B"]],
        on=["match_id", "away_team_id"],
        how="left"
    )


    # 5. Rename goal columns

    final = final.rename(columns={
        "home_score": "Goals A",
        "away_score": "Goals B",
        "date": "Date"
    })


    # 6. Select only the columns we need

    final = final[
        [
            "match_id",
            "Date",
            "Team A",
            "Team B",
            "Goals A",
            "Goals B",
            "Possession A",
            "Possession B"
        ]
    ]

    # 7. Check the result

    print("\nFirst 10 matches:")
    print(final.head(10))

    print("\nNumber of rows:", len(final))

    print("\nMissing values:")
    print(final.isnull().sum())

    # 8. Save the new CSV

    final.to_csv(
        OUTPUTS / "worldcup_2026_possession.csv",
        index=False
    )

    print("\nCreated: worldcup_2026_possession.csv")
    # A decisive score is required to assign winning and losing possession.
    no_draws = final[final["Goals A"] != final["Goals B"]].copy()
    no_draws.to_csv(OUTPUTS / "matches_no_draws.csv", index=False)
    print("Matches after removing draws:", len(no_draws))
    return no_draws


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    matches = build_dataset()
    results = calculate_statistics(matches, save_dir=OUTPUTS)
    (OUTPUTS / "statistical_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
