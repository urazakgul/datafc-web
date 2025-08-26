import streamlit as st
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

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

    xg_by_team_sit_long = (
        shots_df.groupby(["team_name", "situation"], as_index=False)
                .agg(xg_sum=("xg", "sum"))
    )

    team_totals = (
        xg_by_team_sit_long.groupby("team_name", as_index=False)["xg_sum"]
                           .sum()
                           .rename(columns={"xg_sum": "xg_total"})
    )

    xg_share_long = xg_by_team_sit_long.merge(team_totals, on="team_name")
    xg_share_long["xg_pct"] = xg_share_long["xg_sum"] / xg_share_long["xg_total"]

    xg_share_wide = (
        xg_share_long.pivot(index="team_name", columns="situation", values="xg_pct")
                     .fillna(0)
                     .applymap(lambda v: round(v * 100, 1))
    )

    values = xg_share_wide.values.astype(float)
    teams = xg_share_wide.index.tolist()
    sits  = xg_share_wide.columns.tolist()

    n_rows, n_cols = values.shape
    cmap = plt.get_cmap("Reds")

    global_vmin = float(np.min(values))
    global_vmax = float(np.max(values))
    if np.isclose(global_vmin, global_vmax):
        global_vmax = global_vmin + 1e-6
    norm = Normalize(vmin=global_vmin, vmax=global_vmax)

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
            ax.text(j, i, f"{values[i, j]:.1f}%", va="center", ha="center", fontsize=10)

    ax.grid(False)
    ax.set_axisbelow(False)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"{season} {league}\nExpected Goals (xG) Contribution of Teams by Situation\n(up to Week {max_week})",
        fontsize=14,
        fontweight="bold",
        pad=40
    )

    st.pyplot(fig)