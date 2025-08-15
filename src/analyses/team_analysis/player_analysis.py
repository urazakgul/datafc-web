import streamlit as st
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from adjustText import adjust_text
import itertools
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import itertools

plt.style.use("fivethirtyeight")

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

EVENT_COLORS = {
    "pass": "#0074D9",
    "goal": "#FF4136",
    "free-kick": "#2ECC40",
    "clearance": "#B10DC9",
    "ball-movement": "#FF851B",
    "corner": "#FFDC00",
    "post": "#FF69B4",
    "save": "#7FDBFF",
    "miss": "#AAAAAA"
}

def _get_dynamic_color_map(event_types, static_map=EVENT_COLORS):
    dynamic_map = {k: to_hex(v) for k, v in static_map.items()}
    used_colors = set(dynamic_map.values())

    palette = itertools.cycle(plt.get_cmap("tab20").colors)

    for ev in sorted(event_types):
        if ev not in dynamic_map:
            candidate = to_hex(next(palette))
            while candidate in used_colors:
                candidate = to_hex(next(palette))
            dynamic_map[ev] = candidate
            used_colors.add(candidate)

    return dynamic_map

def run(team: str, country: str, league: str, season: str):
    selected_player_analysis = st.selectbox(
        "Breakdown by:",
        [
            "Goal Sequence Involvement",
            "Expected Goals vs Expected Assists",
            "Rating Consistency & Average"
        ],
        index=None,
        placeholder="Please select an analysis type"
    )

    if selected_player_analysis:

        requirements = {
            "Goal Sequence Involvement": ["match_data", "shots_data", "goal_networks_data"],
            "Expected Goals vs Expected Assists": ["match_data", "lineups_data"],
            "Rating Consistency & Average": ["match_data", "lineups_data"],
        }

        required_keys = requirements[selected_player_analysis]

        dfs_tuple = require_session_data(*required_keys)
        data = {k: v for k, v in zip(required_keys, dfs_tuple)}

        match_df = data["match_data"]

        max_week = match_df["week"].max()

    if selected_player_analysis == "Goal Sequence Involvement":
        shots_df = data["shots_data"]
        goal_networks_df = data["goal_networks_data"]
        def merge_match_data(match_df, shots_df):
            filtered_shots = shots_df[shots_df["shot_type"] == "goal"][
                ["tournament", "season", "week", "game_id", "player_name", "is_home", "goal_type", "xg"]
            ]
            merged_df = match_df.merge(filtered_shots, on=["tournament", "season", "week", "game_id"])
            return merged_df[~merged_df["goal_type"].isin(["penalty", "own"])]

        match_df = filter_matches_by_status(match_df, "Ended")
        match_df = match_df[["tournament", "season", "week", "game_id", "home_team", "away_team"]]
        match_shots_df = merge_match_data(match_df, shots_df)

        goal_networks_scored_df = goal_networks_df.copy()
        goal_networks_scored_df["team_name"] = None

        for game_id in match_shots_df["game_id"].unique():
            match_data = match_shots_df[match_shots_df["game_id"] == game_id]
            for _, row in match_data.iterrows():
                team_name = row["home_team"] if row["is_home"] else row["away_team"]
                goal_networks_scored_df.loc[
                    (goal_networks_scored_df["game_id"] == game_id) &
                    (goal_networks_scored_df["player_name"] == row["player_name"]) &
                    (goal_networks_scored_df["event_type"] == "goal"), "team_name"
                ] = team_name

        def fill_team_name(df):
            df["team_name"] = df.groupby("id")["team_name"].transform(lambda x: x.ffill().bfill())
            return df

        goal_networks_scored_df = fill_team_name(goal_networks_scored_df)

        for _, group in goal_networks_scored_df.groupby("id"):
            if (group["event_type"] == "goal").any() and group.loc[group["event_type"] == "goal", "goal_shot_x"].iloc[0] != 100:
                goal_networks_scored_df.loc[group.index, ["player_x", "player_y"]] = 100 - group[["player_x", "player_y"]]

        goal_networks_scored_df = goal_networks_scored_df.merge(match_df, on=["tournament", "season", "week", "game_id"])
        side_data = goal_networks_scored_df[goal_networks_scored_df["team_name"] == team]

        if side_data.empty:
            st.warning(f"No data available yet for {team} in {season} {league}.")
            return

        player_event_counts = side_data.groupby(["player_name", "event_type"]).size().reset_index(name="count")
        player_total_counts = player_event_counts.groupby("player_name")["count"].sum().reset_index(name="total_count")

        player_event_counts = player_event_counts.merge(player_total_counts, on="player_name")
        player_event_counts["percentage"] = (player_event_counts["count"] / player_event_counts["total_count"]) * 100

        pivot_df = player_event_counts.pivot(index="player_name", columns="event_type", values="count").fillna(0)
        pivot_df["total_count"] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values(by="total_count")

        event_columns = [col for col in pivot_df.columns if col != "total_count"]
        event_colors = _get_dynamic_color_map(event_columns, EVENT_COLORS)
        colors = [event_colors[event] for event in event_columns]

        fig, ax = plt.subplots(figsize=(16, 16))
        pivot_df[event_columns].plot(kind="barh", stacked=True, ax=ax, color=colors)

        total_column = "total_count"
        stat_columns = event_columns

        for i, (index, row) in enumerate(pivot_df.iterrows()):
            total = row[total_column]
            if total > 0:
                start = 0
                for col in stat_columns:
                    percent = (row[col] / total * 100) if row[col] > 0 else 0
                    if percent > 0:
                        ax.text(start + row[col] / 2, i, f"%{percent:.0f}", ha="center", va="center", fontsize=9, color="black")
                    start += row[col]

        ax.set_xlabel("Number of Involvements in Goal Sequences", labelpad=20)
        ax.set_ylabel("")
        ax.set_title(
            f"{season} {league}\n Goal Sequence Involvement for {team}\n(up to Week {max_week})",
            fontsize=18,
            fontweight="bold",
            pad=70
        )

        custom_legend = [
            Line2D([0], [0], color=event_colors[event], lw=8, label=event.capitalize())
            for event in event_columns
        ]
        ax.legend(
            handles=custom_legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(custom_legend),
            fontsize=12,
            frameon=False
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune="lower", nbins=6))
        ax.grid(True, linestyle="--", alpha=0.7)

        st.pyplot(fig)

    elif selected_player_analysis == "Expected Goals vs Expected Assists":
        lineups_df = data["lineups_data"]

        match_df = filter_matches_by_status(match_df, "Ended")
        match_df = match_df[[
            "country","tournament","season","week","game_id",
            "home_team","home_team_id","away_team","away_team_id"
        ]]

        lineups_df = lineups_df[[
            "country","tournament","season","week","game_id","team","player_id","player_name",
            "expectedassists","expectedgoals"
        ]]

        merged_df = lineups_df.merge(
            match_df,
            on=["country", "tournament", "season", "week", "game_id"],
            how="left"
        )
        merged_df["team_name"] = merged_df.apply(
            lambda row: row["home_team"] if row["team"] == "home" else row["away_team"],
            axis=1
        )

        filtered_df = merged_df[merged_df["team_name"] == team]

        if filtered_df.empty:
            st.warning(f"No data available yet for {team} in {season} {league}.")
            return

        agg_df = filtered_df.groupby("player_name", as_index=False)[["expectedgoals", "expectedassists"]].sum()

        xg_75 = np.percentile(agg_df["expectedgoals"], 75)
        xa_75 = np.percentile(agg_df["expectedassists"], 75)

        xg_norm = (agg_df["expectedgoals"] - agg_df["expectedgoals"].min()) / (agg_df["expectedgoals"].max() - agg_df["expectedgoals"].min() + 1e-6)
        xa_norm = (agg_df["expectedassists"] - agg_df["expectedassists"].min()) / (agg_df["expectedassists"].max() - agg_df["expectedassists"].min() + 1e-6)

        colors = [(r, 0.2, b) for r, b in zip(xg_norm, xa_norm)]

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(agg_df["expectedgoals"], agg_df["expectedassists"], s=80, color=colors, alpha=0.85)

        texts = []
        for i, row in agg_df.iterrows():
            if (row["expectedgoals"] >= xg_75) or (row["expectedassists"] >= xa_75):
                texts.append(
                    ax.text(
                        row["expectedgoals"], row["expectedassists"],
                        row["player_name"], fontsize=10, ha='left', va='bottom', fontweight="bold"
                    )
                )
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

        ax.set_title(
            f"{season} {league}\n{team} Players - Expected Goals vs Expected Assists\n(Showing Top 25%, up to Week {max_week})",
            fontsize=13,
            fontweight="bold",
            pad=30
        )
        ax.set_xlabel("Expected Goals (xG)", labelpad=20)
        ax.set_ylabel("Expected Assists (xA)", labelpad=20)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.axvline(x=xg_75, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.axhline(y=xa_75, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        st.pyplot(fig)

    elif selected_player_analysis == "Rating Consistency & Average":
        lineups_df = data["lineups_data"]

        match_df = filter_matches_by_status(match_df, "Ended")
        match_df = match_df[[
            "country","tournament","season","week","game_id",
            "home_team","home_team_id","away_team","away_team_id"
        ]]

        lineups_df = lineups_df[[
            "country","tournament","season","week","game_id","team","player_id","player_name","rating"
        ]]

        merged_df = lineups_df.merge(
            match_df,
            on=["country", "tournament", "season", "week", "game_id"],
            how="left"
        )
        merged_df["team_name"] = merged_df.apply(
            lambda row: row["home_team"] if row["team"] == "home" else row["away_team"],
            axis=1
        )
        filtered_df = merged_df[merged_df["team_name"] == team]

        if filtered_df.empty:
            st.warning(f"No data available yet for {team} in {season} {league}.")
            return

        weeks = pd.to_numeric(filtered_df["week"], errors="coerce").dropna().unique()
        if len(weeks) < 2:
            st.info(f"'Rating Consistency & Average' analysis requires data from at least two different weeks.")
            return

        agg_df = (
            filtered_df
            .groupby("player_name", as_index=False)
            .agg(rating_mean=("rating", "mean"), rating_std=("rating", "std"))
        ).dropna()

        mean_norm = (agg_df["rating_mean"] - agg_df["rating_mean"].min()) / (agg_df["rating_mean"].max() - agg_df["rating_mean"].min() + 1e-6)
        std_norm = (agg_df["rating_std"] - agg_df["rating_std"].min()) / (agg_df["rating_std"].max() - agg_df["rating_std"].min() + 1e-6)
        colors = [(r, 0.2, b) for r, b in zip(mean_norm, std_norm)]

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(agg_df["rating_mean"], agg_df["rating_std"], s=80, color=colors, alpha=0.85)

        for i, row in agg_df.iterrows():
            ax.text(
                row["rating_mean"], row["rating_std"],
                row["player_name"], fontsize=10, ha='left', va='bottom'
            )

        adjust_text(ax.texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

        ax.set_title(
            f"{season} {league}\n{team} Player Rating Consistency & Average\n(up to Week {max_week})",
            fontsize=13, fontweight="bold", pad=30
        )
        ax.set_xlabel("Rating Average (higher is better)", labelpad=20)
        ax.set_ylabel("Rating Consistency (Std. Deviation, lower is better)", labelpad=20)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        ax.axvline(x=7, color='crimson', linestyle='-', linewidth=2, alpha=0.6, label='Good Performance (7.0)')
        ax.legend(loc='upper right', fontsize=11, frameon=False)

        ax.invert_yaxis()

        st.pyplot(fig)