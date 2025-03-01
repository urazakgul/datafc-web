import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def calculate_mean_distance(player_positions):
    distances = []
    for i, player1 in player_positions.iterrows():
        for j, player2 in player_positions.iterrows():
            if i != j:
                distance = np.sqrt((player1["x"] - player2["x"])**2 + (player1["y"] - player2["y"])**2)
                distances.append(distance)
    return np.mean(distances) if distances else None

def calculate_horizontal_vertical_spread(player_positions):
    horizontal_spread = player_positions["x"].std()
    vertical_spread = player_positions["y"].std()
    return horizontal_spread, vertical_spread

def create_geometry_plot(overall_data, category, last_round):
    if category == "Compactness":
        metric = "mean_distance"
        xlabel = "Compactness (Average)"
    elif category == "Horizontal Spread":
        metric = "horizontal_spread"
        xlabel = "Horizontal Spread (Standard Deviation)"
    elif category == "Vertical Spread":
        metric = "vertical_spread"
        xlabel = "Vertical Spread (Standard Deviation)"

    team_means = overall_data.groupby("team_name")[metric].mean().sort_values(ascending=False)
    sorted_teams = team_means.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 12))
    for team in sorted_teams:
        team_data = overall_data[overall_data["team_name"] == team]

        team_min = team_data[metric].min()
        team_max = team_data[metric].max()
        team_mean = team_data[metric].mean()

        ax.scatter(
            team_data[metric], [team] * len(team_data), color='grey', edgecolors='black', alpha=0.7, s=50
        )

        ax.hlines(y=team, xmin=team_min, xmax=team_max, colors='black', linestyles='-', linewidth=0.5, alpha=0.7)

        ax.scatter(
            [team_min], [team], color="blue", edgecolors="black", alpha=0.7, s=100,
            label="Minimum" if team == sorted_teams[0] else "", zorder=3
        )
        ax.scatter(
            [team_mean], [team], color="orange", edgecolors="black", alpha=0.7, s=100,
            label="Average" if team == sorted_teams[0] else "", zorder=3
        )
        ax.scatter(
            [team_max], [team], color="red", edgecolors="black", alpha=0.7, s=100,
            label="Maximum" if team == sorted_teams[0] else "", zorder=3
        )

    ax.set_yticks(range(len(sorted_teams)))
    ax.set_yticklabels(sorted_teams, fontsize=9)
    ax.set_title(
        f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - {category} Across Teams",
        fontsize=16,
        fontweight="bold",
        pad=30
    )
    ax.set_xlabel(xlabel, fontsize=12, labelpad=15)
    ax.set_ylabel("")
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=8
    )
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    add_footer(fig, y=-0.01)
    st.pyplot(fig)

def main(category):
    try:

        match_data_df = get_data("match_data")
        coordinates_data_df = get_data("coordinates_data")
        lineups_data_df = get_data("lineups_data")
        substitutions_data_df = get_data("substitutions_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]
        match_data_df = match_data_df[["tournament","season","week","game_id","home_team","away_team"]]
        coordinates_data_df = coordinates_data_df.groupby(["tournament","season","week","game_id","player_name", "player_id"]).agg({'x': 'mean', 'y': 'mean'}).reset_index()
        lineups_data_df = lineups_data_df[["tournament","season","week","game_id","team","player_name", "player_id"]].drop_duplicates()

        final_merged_data = pd.merge(
            pd.merge(
                coordinates_data_df,
                lineups_data_df,
                on=["tournament", "season", "week", "game_id", "player_name", "player_id"]
            ),
            match_data_df,
            on=["tournament", "season", "week", "game_id"]
        )
        final_merged_data["team_name"] = final_merged_data.apply(lambda row: row["home_team"] if row["team"] == "home" else row["away_team"], axis=1)
        final_merged_data = final_merged_data.drop(columns=["team","home_team","away_team"])

        def check_player_status_and_time(row):
            player_in = substitutions_data_df[
                (substitutions_data_df["tournament"] == row["tournament"]) &
                (substitutions_data_df["season"] == row["season"]) &
                (substitutions_data_df["week"] == row["week"]) &
                (substitutions_data_df["game_id"] == row["game_id"]) &
                (substitutions_data_df["player_in"] == row["player_name"]) &
                (substitutions_data_df["player_in_id"] == row["player_id"])
            ]
            player_out = substitutions_data_df[
                (substitutions_data_df["tournament"] == row["tournament"]) &
                (substitutions_data_df["season"] == row["season"]) &
                (substitutions_data_df["week"] == row["week"]) &
                (substitutions_data_df["game_id"] == row["game_id"]) &
                (substitutions_data_df["player_out"] == row["player_name"]) &
                (substitutions_data_df["player_out_id"] == row["player_id"])
            ]

            status = "Starting 11"
            time = None
            if not player_in.empty:
                status = "Subbed in"
                time = player_in['time'].iloc[0]
            elif not player_out.empty:
                status = "Subbed out"
                time = player_out['time'].iloc[0]

            return status, time

        final_merged_data[['status', 'time']] = final_merged_data.apply(
            lambda row: pd.Series(check_player_status_and_time(row)), axis=1
        )

        if category in ["Compactness", "Vertical Spread", "Horizontal Spread"]:
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
                        initial_horizontal_spread, initial_vertical_spread = calculate_horizontal_vertical_spread(team_data)
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
                            horizontal_spread, vertical_spread = calculate_horizontal_vertical_spread(team_data)
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

        last_round = match_data_df['week'].max()

        create_geometry_plot(overall_data, category, last_round)

    except Exception as e:
        st.error("No suitable data found.")
        st.markdown(
            """
            <a href="https://github.com/urazakgul/datafc-web/issues" target="_blank" class="error-button">
                🛠️ Report Issue
            </a>
            """,
            unsafe_allow_html=True
        )