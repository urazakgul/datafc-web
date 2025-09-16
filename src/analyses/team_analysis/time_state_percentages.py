import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

def _record_interval(acc, home, away, dt: float, leader: str | None):
    if dt <= 0:
        return
    if leader is None:
        acc[home]["draw"] += dt
        acc[away]["draw"] += dt
    else:
        other = away if leader == home else home
        acc[leader]["lead"] += dt
        acc[other]["behind"] += dt

def _compute_lead_draw_behind(df_match: pd.DataFrame) -> pd.DataFrame:
    home = df_match["home_team"].iloc[0]
    away = df_match["away_team"].iloc[0]
    gid  = df_match["game_id"].iloc[0]

    match_end_min = float(df_match["match_minute"].max())

    goals = (
        df_match.loc[df_match["is_goal"] == 1, ["match_minute", "team_name"]]
        .sort_values("match_minute", kind="mergesort")
        .reset_index(drop=True)
    )

    acc = {
        home: {"lead": 0.0, "behind": 0.0, "draw": 0.0},
        away: {"lead": 0.0, "behind": 0.0, "draw": 0.0},
    }
    score = {home: 0, away: 0}

    last_t = 0.0
    for _, row in goals.iterrows():
        t = float(row["match_minute"])
        leader = None if score[home] == score[away] else (home if score[home] > score[away] else away)
        _record_interval(acc, home, away, t - last_t, leader)

        scorer = row["team_name"]
        score[scorer] += 1
        last_t = t

    leader = None if score[home] == score[away] else (home if score[home] > score[away] else away)
    _record_interval(acc, home, away, match_end_min - last_t, leader)

    rows = []
    for team in (home, away):
        rows.append({
            "game_id": gid,
            "team_name": team,
            "minutes_behind": acc[team]["behind"],
            "minutes_draw":   acc[team]["draw"],
            "minutes_lead":   acc[team]["lead"],
            "match_end_minute": match_end_min,
        })
    return pd.DataFrame(rows)

def _aggregate_team_level(per_match_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        per_match_df
        .groupby("team_name", as_index=False)
        .agg(
            games_played=("game_id", "nunique"),
            total_minutes=("match_end_minute", "sum"),
            minutes_behind=("minutes_behind", "sum"),
            minutes_draw=("minutes_draw", "sum"),
            minutes_lead=("minutes_lead", "sum"),
        )
    )

    for col in ["minutes_behind", "minutes_draw", "minutes_lead"]:
        agg[col.replace("minutes_", "pct_")] = (agg[col] / agg["total_minutes"]).round(4)

    time_cols = ["minutes_behind", "minutes_draw", "minutes_lead", "total_minutes"]
    agg[time_cols] = agg[time_cols].round(2)

    agg = agg.sort_values("pct_lead", ascending=False, kind="mergesort").reset_index(drop=True)
    return agg

def run(country: str, league: str, season: str):
    match_df, shots_df = require_session_data("match_data", "shots_data")

    match_df = filter_matches_by_status(match_df, "Ended")
    match_df = match_df[["season", "week", "game_id", "home_team", "away_team"]]

    max_week = match_df["week"].max()

    shots_df = shots_df.merge(match_df, on=["season", "week", "game_id"], how="left")

    shots_df["team_name"] = shots_df.apply(
        lambda row: row["home_team"] if row.get("is_home", False) else row["away_team"],
        axis=1
    )
    shots_df["is_goal"] = shots_df["shot_type"].apply(lambda x: 1 if x == "goal" else 0)
    shots_df["match_minute"] = shots_df["time"].fillna(0).astype(float) + shots_df["added_time"].fillna(0).astype(float)

    per_match = (
        shots_df.groupby("game_id", group_keys=False)
        .apply(_compute_lead_draw_behind)
        .reset_index(drop=True)
    )

    team_level = _aggregate_team_level(per_match)

    team_level = team_level.sort_values("pct_lead", ascending=False, kind="mergesort").reset_index(drop=True)
    teams = team_level["team_name"].tolist()

    lead_vals = (team_level["pct_lead"] * 100).round(1).values
    draw_vals = (team_level["pct_draw"] * 100).round(1).values
    behind_vals = (team_level["pct_behind"] * 100).round(1).values

    fig, axes = plt.subplots(1, 3, figsize=(32, max(20, 1.0 * len(teams))))

    def _plot_subplot(ax, values, title, color):
        bars = ax.barh(teams, values, color=color)
        ax.set_title(title, fontsize=24, fontweight="bold")
        ax.tick_params(axis="x", labelsize=24)
        ax.tick_params(axis="y", labelsize=24)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.7)
        ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%",
                va='center',
                ha='left',
                fontsize=20
            )

    _plot_subplot(axes[0], lead_vals, "Winning", "#2ca02c")
    _plot_subplot(axes[1], draw_vals, "Drawing", "#7f7f7f")
    _plot_subplot(axes[2], behind_vals, "Losing", "#d62728")

    plt.suptitle(
        f"{season} {league}\n"
        f"Percentage of Total Time Spent in Each Game State\n"
        f"(up to Week {max_week})",
        fontsize=36,
        fontweight="bold",
        y=1.05
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)