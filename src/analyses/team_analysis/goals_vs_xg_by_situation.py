import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.style.use("fivethirtyeight")

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

def run(country: str, league: str, season: str):
    match_df, shots_df = require_session_data("match_data", "shots_data")

    match_df = filter_matches_by_status(match_df, "Ended")
    match_df = match_df[["season", "week", "game_id", "home_team", "away_team"]]
    max_week = match_df["week"].max()

    shots_df = shots_df.merge(
        match_df,
        on=["season", "week", "game_id"],
        how="left"
    )

    shots_df["team_name"] = shots_df.apply(
        lambda row: row["home_team"] if row["is_home"] else row["away_team"], axis=1
    )
    shots_df["is_goal"] = shots_df["shot_type"].apply(lambda x: 1 if x == "goal" else 0)
    shots_df = shots_df[shots_df["goal_type"] != "own"]

    agg_long = (
        shots_df.groupby(["team_name", "situation"], as_index=False)
                .agg(
                    xg_sum=("xg", "sum"),
                    goals=("is_goal", "sum")
                )
    )

    agg_long["goals_minus_xg"] = agg_long["goals"] - agg_long["xg_sum"]

    diff_wide = (
        agg_long.pivot(index="team_name", columns="situation", values="goals_minus_xg")
               .fillna(0.0)
    )

    values = diff_wide.values.astype(float)
    teams = diff_wide.index.tolist()
    sits  = diff_wide.columns.tolist()

    n_rows, n_cols = values.shape

    cmap = plt.get_cmap("coolwarm_r")

    vmin, vmax = np.min(values), np.max(values)
    span = max(abs(vmin), abs(vmax))
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

    rgba = cmap(norm(values))

    fig_w = max(8, n_cols * 0.8)
    fig_h = max(6, n_rows * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")

    ax.imshow(rgba, aspect="auto", interpolation="nearest", origin="upper")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(sits, rotation=35, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(teams)

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f"{values[i, j]:+,.1f}", va="center", ha="center", fontsize=10)

    ax.grid(False)
    ax.set_axisbelow(False)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"{season} {league}\nGoals vs Expected Goals (xG) Difference by Situation\n(up to Week {max_week})",
        fontsize=14,
        fontweight="bold",
        pad=40
    )

    st.pyplot(fig)