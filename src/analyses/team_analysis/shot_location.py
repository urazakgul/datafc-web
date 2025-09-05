import streamlit as st
import numpy as np
from mplsoccer import VerticalPitch
from matplotlib.lines import Line2D

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

def _exclude_own_goals(df):
    df = df.copy()
    df = df[df["goal_type"].str.lower() != "own"]
    return df

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

def _standardize_right_attack(df, x_col="player_coordinates_x", y_col="player_coordinates_y"):
    df[x_col] = 100 - df[x_col]
    df[y_col] = 100 - df[y_col]
    return df

def _compute_shot_stats(df):
    df = df.copy()
    df["is_goal"] = (df.get("shot_type") == "goal").astype(int)

    total_shots = len(df)
    total_goals = int(df["is_goal"].sum())
    total_xg = float(df["xg"].sum()) if "xg" in df.columns else 0.0

    non_penalty = df[df.get("situation") != "penalty"]
    non_penalty_goals = int(non_penalty["is_goal"].sum())
    non_penalty_xg = float(non_penalty["xg"].sum()) if "xg" in non_penalty.columns else 0.0

    shot_conversion = _conv_rate(total_goals, total_shots)
    non_penalty_conversion = _conv_rate(non_penalty_goals, len(non_penalty))

    return {
        "total_shots": total_shots,
        "total_goals": total_goals,
        "total_xg": total_xg,
        "np_goals": non_penalty_goals,
        "np_xg": non_penalty_xg,
        "conv": shot_conversion,
        "np_conv": non_penalty_conversion
    }

def _draw_shot_map(df, title, max_week):
    pitch = VerticalPitch(
        pitch_type="opta",
        half=False,
        label=False,
        tick=False,
        corner_arcs=True
    )
    fig, ax = pitch.draw(figsize=(7, 10))

    df = df.copy()
    df["is_goal"] = (df["shot_type"] == "goal").astype(int)
    goals_df = df[df["is_goal"] == 1]
    missed_df = df[df["is_goal"] == 0]

    series = [
        ("Goal", "red", goals_df, 3),
        ("No Goal", "gray", missed_df, 2),
    ]
    for label, color, df_, z in series:
        if not df_.empty:
            _plot_points(pitch, ax, df_, label, color, z)

    ax.set_title(f"{title}\n(up to Week {max_week})", fontsize=10, fontweight="bold", loc="center", pad=55)

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

    return fig, ax

def run(team: str, country: str, league: str, season: str):
    match_df, shots_df = require_session_data("match_data", "shots_data")
    match_df = filter_matches_by_status(match_df, "Ended")

    match_cols = ["tournament", "season", "week", "game_id", "home_team", "away_team"]
    shots_df = shots_df.merge(match_df[match_cols], on=["tournament", "season", "week", "game_id"], how="left")

    max_week = match_df["week"].max()

    is_home = shots_df.get("is_home")
    if is_home is None:
        is_home = np.zeros(len(shots_df), dtype=bool)
    else:
        is_home = is_home.fillna(False).astype(bool)

    shots_df["team_name"] = np.where(is_home, shots_df["home_team"], shots_df["away_team"])
    shots_df["opponent_team_name"] = np.where(is_home, shots_df["away_team"], shots_df["home_team"])

    team_shots_for = shots_df.loc[shots_df["team_name"] == team].copy()
    team_shots_for = _exclude_own_goals(team_shots_for)

    team_shots_against = shots_df.loc[shots_df["opponent_team_name"] == team].copy()
    team_shots_against = _exclude_own_goals(team_shots_against)

    if team_shots_for.empty and team_shots_against.empty:
        st.warning(f"No data available yet for {team} in {season} {league}.")
        return

    info_box = st.empty()
    info_box.info("Two separate charts will be displayed below shortly.")

    if not team_shots_for.empty:
        team_shots_for = _standardize_right_attack(team_shots_for)
    if not team_shots_against.empty:
        team_shots_against = _standardize_right_attack(team_shots_against)

    if not team_shots_for.empty:
        stats_for = _compute_shot_stats(team_shots_for)
        fig_for, ax_for = _draw_shot_map(
            team_shots_for,
            title=f"{season} {league}\nShot Locations by {team}",
            max_week=max_week
        )
        ax_for.text(
            50, 108,
            (
                f"Total Shots: {stats_for['total_shots']} | Goals: {stats_for['total_goals']} (excl. own goals & awarded)\n"
                f"xG: {stats_for['total_xg']:.2f} | NP-xG: {stats_for['np_xg']:.2f}\n"
                f"Conversion: %{stats_for['conv']:.1f} | Non-Penalty Conversion: %{stats_for['np_conv']:.1f}"
            ),
            ha="center", va="center", fontsize=9, color="gray"
        )
        st.pyplot(fig_for)

    if not team_shots_against.empty:
        stats_against = _compute_shot_stats(team_shots_against)
        fig_ag, ax_ag = _draw_shot_map(
            team_shots_against,
            title=f"{season} {league}\nShot Locations Against {team}",
            max_week=max_week
        )

        ax_ag.text(
            50, 108,
            (
                f"Total Shots: {stats_against['total_shots']} | Goals: {stats_against['total_goals']} (excl. own goals & awarded)\n"
                f"xG: {stats_against['total_xg']:.2f} | NP-xG: {stats_against['np_xg']:.2f}\n"
                f"Conversion: %{stats_against['conv']:.1f} | Non-Penalty Conversion: %{stats_against['np_conv']:.1f}"
            ),
            ha="center", va="center", fontsize=9, color="gray"
        )
        st.pyplot(fig_ag)

    if (not team_shots_for.empty) and (not team_shots_against.empty):
        info_box.success(f"Both 'by' and 'Against' shot maps for {team} are displayed below.")
    elif not team_shots_for.empty:
        info_box.success(f"Only 'by' shot map for {team} is available. No data for 'Against'.")
    elif not team_shots_against.empty:
        info_box.success(f"Only 'Against' shot map for {team} is available. No data for 'by'.")