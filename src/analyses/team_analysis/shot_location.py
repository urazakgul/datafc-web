import numpy as np
import streamlit as st
from mplsoccer import VerticalPitch
from matplotlib.lines import Line2D

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

def _conv_rate(goals: int, shots: int) -> float:
    return (goals / shots * 100) if shots else 0.0

def _plot_points(pitch, ax, df, label, color, zorder):
    return pitch.scatter(
        x=df["player_coordinates_x"],
        y=df["player_coordinates_y"],
        ax=ax,
        color=color,
        edgecolors=color,
        s=80,
        zorder=zorder,
        label=label,
        alpha=0.5,
        marker="h"
    )

def run(team: str, country: str, league: str, season: str):
    match_df, shots_df = require_session_data("match_data", "shots_data")
    match_df = filter_matches_by_status(match_df, "Ended")

    max_week = match_df["week"].max()

    shots_df = shots_df.merge(
        match_df[["tournament", "season", "week", "game_id", "home_team", "away_team"]],
        on=["tournament", "season", "week", "game_id"],
        how="left"
    )

    is_home = shots_df.get("is_home")
    if is_home is None:
        is_home = np.zeros(len(shots_df), dtype=bool)
    else:
        is_home = is_home.fillna(False).astype(bool)

    shots_df["team_name"] = np.where(is_home, shots_df["home_team"], shots_df["away_team"])

    team_shots = shots_df.loc[shots_df["team_name"] == team].copy()

    if "goal_type" in team_shots.columns:
        team_shots = team_shots[team_shots["goal_type"] != "own"]

    if team_shots.empty:
        st.warning(f"No data available yet for {team} in {season} {league}.")
        return

    for col in ("player_coordinates_x", "player_coordinates_y"):
        team_shots[col] = 100 - team_shots[col]

    team_shots["is_goal"] = (team_shots["shot_type"] == "goal").astype(int)

    total_shots = len(team_shots)
    total_goals = int(team_shots["is_goal"].sum())
    shot_conversion = _conv_rate(total_goals, total_shots)

    non_penalty = team_shots[team_shots["situation"] != "penalty"]
    non_penalty_goals = int(non_penalty["is_goal"].sum())
    non_penalty_conversion = _conv_rate(non_penalty_goals, len(non_penalty))
    non_penalty_xg = float(non_penalty["xg"].sum()) if "xg" in non_penalty.columns else 0.0

    total_xg = float(team_shots["xg"].sum()) if "xg" in team_shots.columns else 0.0

    pitch = VerticalPitch(
        pitch_type="opta",
        half=False,
        label=False,
        tick=False,
        corner_arcs=True
    )
    fig, ax = pitch.draw(figsize=(7, 10))

    goals_df = team_shots[team_shots["is_goal"] == 1]
    missed_df = team_shots[team_shots["is_goal"] == 0]
    series = [
        ("Goal Scored", "red", goals_df, 3),
        ("No Goal", "gray", missed_df, 2),
    ]
    for label, color, df_, z in series:
        if not df_.empty:
            _plot_points(pitch, ax, df_, label, color, z)

    title = f"{season} {league}\nShot Locations for {team}\n(up to Week {max_week})"
    ax.set_title(title, fontsize=10, fontweight="bold", loc="center", pad=55)

    ax.text(
        50, 108,
        (
            f"Total Shots: {total_shots} | Goals: {total_goals} (excl. own goals & awarded matches)\n"
            f"xG: {total_xg:.2f} | NP-xG: {non_penalty_xg:.2f}\n"
            f"Conversion: %{shot_conversion:.1f} | Non-Penalty Conversion: %{non_penalty_conversion:.1f}"
        ),
        ha="center",
        va="center",
        fontsize=9,
        color="gray"
    )

    custom_legend = [
        Line2D(
            [0], [0],
            marker="h",
            linestyle="None",
            label=label,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=10
        )
        for label, color, _, _ in series
    ]
    ax.legend(
        handles=custom_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=8
    )

    st.pyplot(fig)