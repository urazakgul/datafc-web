import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.components.explanations import explanations
from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

def calculate_mean_distance(player_positions):
    distances = []
    for i, player1 in player_positions.iterrows():
        for j, player2 in player_positions.iterrows():
            if i != j:
                distance = np.sqrt((player1["mean_x"] - player2["mean_x"])**2 + (player1["mean_y"] - player2["mean_y"])**2)
                distances.append(distance)
    return np.mean(distances) if distances else None

def calculate_horizontal_vertical_spread(player_positions):
    horizontal_spread = player_positions["mean_x"].std()
    vertical_spread = player_positions["mean_y"].std()
    return horizontal_spread, vertical_spread

@st.cache_data(show_spinner="Calculating geometric metrics...")
def compute_geometric_analysis(match_df, coordinates_df, lineups_df, substitutions_df):
    match_df = filter_matches_by_status(match_df, "Ended")
    match_df = match_df[["tournament","season","week","game_id","home_team","away_team"]]

    final_merged_data = pd.merge(
        pd.merge(
            coordinates_df,
            lineups_df,
            on=["tournament", "season", "week", "game_id", "team", "player_name", "player_id"]
        ),
        match_df,
        on=["tournament", "season", "week", "game_id"]
    )

    final_merged_data["team_name"] = final_merged_data.apply(lambda row: row["home_team"] if row["team"] == "home" else row["away_team"], axis=1)
    final_merged_data = final_merged_data.drop(columns=["team","home_team","away_team"])

    def check_player_status_and_time(row):
        player_in = substitutions_df[
            (substitutions_df["tournament"] == row["tournament"]) &
            (substitutions_df["season"] == row["season"]) &
            (substitutions_df["week"] == row["week"]) &
            (substitutions_df["game_id"] == row["game_id"]) &
            (substitutions_df["player_in"] == row["player_name"]) &
            (substitutions_df["player_in_id"] == row["player_id"])
        ]
        player_out = substitutions_df[
            (substitutions_df["tournament"] == row["tournament"]) &
            (substitutions_df["season"] == row["season"]) &
            (substitutions_df["week"] == row["week"]) &
            (substitutions_df["game_id"] == row["game_id"]) &
            (substitutions_df["player_out"] == row["player_name"]) &
            (substitutions_df["player_out_id"] == row["player_id"])
        ]

        status = "Starting 11"
        time = None
        if not player_in.empty:
            status = "Subbed in"
            time = player_in["time"].iloc[0]
        elif not player_out.empty:
            status = "Subbed out"
            time = player_out["time"].iloc[0]

        return status, time

    final_merged_data[["status", "time"]] = final_merged_data.apply(
        lambda row: pd.Series(check_player_status_and_time(row)), axis=1
    )

    results = []
    for (tournament, season, round_, game_id), group in final_merged_data.groupby(["tournament", "season", "week", "game_id"]):
        team_groups = group.groupby("team_name")
        for team_name, team_data in team_groups:
            time_points = team_data["time"].dropna().sort_values().unique().tolist()
            max_time = max(time_points + [90])
            time_points = [0] + time_points + [max_time]

            active_players = team_data[(team_data["status"] == "Starting 11") | (team_data["status"] == "Subbed out")].copy()

            if len(active_players) > 1:
                initial_mean_distance = calculate_mean_distance(active_players)
                initial_horizontal_spread, initial_vertical_spread = calculate_horizontal_vertical_spread(active_players)
                results.append({
                    "tournament": tournament,
                    "season": season,
                    "week": round_,
                    "game_id": game_id,
                    "team_name": team_name,
                    "start_time": 0,
                    "end_time": time_points[1],
                    "mean_distance": initial_mean_distance,
                    "horizontal_spread": initial_horizontal_spread,
                    "vertical_spread": initial_vertical_spread,
                    "active_players": active_players["player_name"].tolist()
                })

            for start, end in zip(time_points[:-1], time_points[1:]):
                for _, substitution in team_data[(team_data["time"] > start) & (team_data["time"] <= end)].iterrows():
                    if substitution["status"] == "Subbed in":
                        active_players = pd.concat([active_players, substitution.to_frame().T], ignore_index=True)
                    elif substitution["status"] == "Subbed out":
                        active_players = active_players[active_players["player_id"] != substitution["player_id"]]

                if len(active_players) > 1:
                    mean_distance = calculate_mean_distance(active_players)
                    horizontal_spread, vertical_spread = calculate_horizontal_vertical_spread(active_players)
                    player_names = active_players["player_name"].tolist()
                    results.append({
                        "tournament": tournament,
                        "season": season,
                        "week": round_,
                        "game_id": game_id,
                        "team_name": team_name,
                        "start_time": start,
                        "end_time": end,
                        "mean_distance": mean_distance,
                        "horizontal_spread": horizontal_spread,
                        "vertical_spread": vertical_spread,
                        "active_players": player_names
                    })

    data = pd.DataFrame(results)
    overall_data = data.groupby(["tournament", "season", "week", "game_id", "team_name"]).agg({
        "mean_distance": "mean",
        "horizontal_spread": "mean",
        "vertical_spread": "mean"
    }).reset_index()

    return overall_data

def run(team: str, country: str, league: str, season: str):
    match_df, coordinates_df, lineups_df, substitutions_df = require_session_data("match_data", "coordinates_data", "lineups_data", "substitutions_data")

    match_df = filter_matches_by_status(match_df, "Ended")
    max_week = match_df["week"].max()

    overall_data = compute_geometric_analysis(match_df, coordinates_df, lineups_df, substitutions_df)

    if team not in overall_data["team_name"].values:
        st.warning(f"No data available yet for {team} in {season} {league}.")
        return

    metric_options = {
        "Mean Distance": "mean_distance",
        "Horizontal Spread": "horizontal_spread",
        "Vertical Spread": "vertical_spread"
    }

    selected_metric_label = st.selectbox(
        "Metric",
        list(metric_options.keys()),
        index=None,
        placeholder="Please select a metric"
    )

    label_to_key = {
        "Mean Distance": "mean_distance",
        "Horizontal Spread": "horizontal_spread",
        "Vertical Spread": "vertical_spread"
    }

    if selected_metric_label:
        key = label_to_key[selected_metric_label]
        st.info(explanations[key])

        selected_metric = metric_options[selected_metric_label]

        team_order = overall_data.groupby("team_name")[selected_metric].median().sort_values(ascending=False).index.tolist()
        data_for_plot = [overall_data[overall_data["team_name"] == team][selected_metric].dropna().values for team in team_order]

        fig, ax = plt.subplots(figsize=(7, 10))
        box = ax.boxplot(
            data_for_plot,
            vert=False,
            patch_artist=True,
            labels=team_order,
            showfliers=False,
            medianprops=dict(color="black", linewidth=2)
        )
        for i, patch in enumerate(box["boxes"]):
            if team_order[i] == team:
                patch.set_facecolor("red")
            else:
                patch.set_facecolor("lightgrey")
        yticklabels = ax.get_yticklabels()
        for i, label in enumerate(yticklabels):
            if team_order[i] == team:
                label.set_color("black")
                label.set_fontweight("bold")
            else:
                label.set_color("grey")

        ax.set_xlabel(selected_metric.replace("_", " ").title(), labelpad=20)
        ax.set_ylabel("")
        ax.set_title(
            f"{season} {league}\n{selected_metric_label} for {team}\n(up to Week {max_week})",
            fontsize=12,
            fontweight="bold",
            loc="center",
            pad=30
        )
        plt.tight_layout()
        ax.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)