import streamlit as st
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

def run(team: str, country: str, league: str, season: str):
    match_df, shots_df, standings_df = require_session_data("match_data", "shots_data", "standings_data")

    match_df = filter_matches_by_status(match_df, "Ended")

    standings_df = standings_df[standings_df["category"] == "Total"][["team_name", "scores_for", "scores_against"]]

    shots_df = shots_df.merge(match_df, on=["tournament", "season", "week", "game_id"], how="inner")
    shots_df["team_name"] = np.where(shots_df["is_home"], shots_df["home_team"], shots_df["away_team"])

    xg_by_team = (
        shots_df.groupby(["team_name", "week"], as_index=False)["xg"]
        .sum()
        .rename(columns={"xg": "total_xg"})
    )

    ha = match_df[[
        "game_id", "week",
        "home_team", "away_team",
        "home_score_display", "away_score_display"
    ]].copy()

    home = ha.rename(columns={
        "home_team": "team_name",
        "home_score_display": "goals"
    })[["game_id", "week", "team_name", "goals"]]

    away = ha.rename(columns={
        "away_team": "team_name",
        "away_score_display": "goals"
    })[["game_id", "week", "team_name", "goals"]]

    goals_long = pd.concat([home, away], ignore_index=True)
    goals_long["goals"] = pd.to_numeric(goals_long["goals"], errors="coerce").fillna(0)

    goal_shots_by_team_long = (
        goals_long.groupby(["team_name", "week"], as_index=False)["goals"]
        .sum()
        .rename(columns={"goals": "week_goal_count"})
    )

    xg_goal_teams = (
        xg_by_team.merge(goal_shots_by_team_long, on=["team_name", "week"], how="inner")
        .sort_values(["team_name", "week"])
        .assign(
            cumulative_total_xg=lambda d: d.groupby("team_name")["total_xg"].cumsum(),
            cumulative_goal_count=lambda d: d.groupby("team_name")["week_goal_count"].cumsum()
        )
        .assign(
            cum_goal_xg_diff=lambda d: d["cumulative_goal_count"] - d["cumulative_total_xg"],
            week=lambda d: pd.to_numeric(d["week"], errors="coerce")
        )
        .dropna(subset=["week"])
    )

    team_df = xg_goal_teams[xg_goal_teams["team_name"] == team]
    if team_df.empty:
        st.warning(f"No data available yet for {team} in {season} {league}.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    x = team_df["week"].to_numpy()
    y = team_df["cum_goal_xg_diff"].to_numpy()

    if len(x) == 1:
        color = "blue" if y[0] >= 0 else "red"
        ax.scatter(x, y, color=color, s=120, alpha=0.6, edgecolor="k", zorder=5)
        w = int(round(float(x[0])))
        ax.set_xlim(w - 0.6, w + 0.6)
        ax.set_xticks([w])
    else:
        ax.fill_between(x, y, 0, where=(y >= 0), color="blue", alpha=0.3, interpolate=True)
        ax.fill_between(x, y, 0, where=(y < 0), color="red", alpha=0.3, interpolate=True)

        segments, colors = [], []
        for i in range(len(x) - 1):
            x0, y0 = x[i], y[i]
            x1, y1 = x[i + 1], y[i + 1]

            if (y0 < 0 <= y1) or (y0 >= 0 > y1):
                x_cross = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
                segments.append([[x0, y0], [x_cross, 0]])
                colors.append("darkred" if y0 < 0 else "darkblue")
                segments.append([[x_cross, 0], [x1, y1]])
                colors.append("darkblue" if y1 > 0 else "darkred")
            else:
                segments.append([[x0, y0], [x1, y1]])
                colors.append("darkblue" if (y0 >= 0 and y1 >= 0) else "darkred")

        lc = LineCollection(segments, colors=colors, linewidths=2)
        ax.add_collection(lc)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    custom_legend = [
        Line2D([0], [0], color="blue", lw=8, alpha=0.3, label="Above Zero: Actual goals > xG"),
        Line2D([0], [0], color="red", lw=8, alpha=0.3, label="Below Zero: Actual goals < xG"),
    ]
    ax.legend(
        handles=custom_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    ax.set_xlabel("Week", labelpad=20)
    ax.set_ylabel("Cumulative (Goals - xG)", labelpad=20)
    ax.set_title(
        f"{season} {league}\nWeekly Cumulative Goals - xG Difference for {team}",
        fontsize=14,
        fontweight="bold",
        pad=30
    )
    ax.grid(True, linestyle="--", alpha=0.7)

    st.pyplot(fig)