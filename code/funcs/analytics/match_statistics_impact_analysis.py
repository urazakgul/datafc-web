import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import match_performance_posneg, PLOT_STYLE

plt.style.use(PLOT_STYLE)

def clean_percent_columns(dataframe, columns_to_check, target_columns):
    for index, row in dataframe.iterrows():
        if any(keyword in row["stat_name"] for keyword in columns_to_check):
            for col in target_columns:
                dataframe.at[index, col] = row[col].replace("%", "").strip()
    return dataframe

def clean_parenthesis_columns(dataframe, columns_to_check, target_columns):
    for index, row in dataframe.iterrows():
        if any(keyword in row["stat_name"] for keyword in columns_to_check):
            for col in target_columns:
                if "(" in row[col]:
                    dataframe.at[index, col] = row[col].split("(")[0].strip()
    return dataframe

def classify_difference(row):
     if (row["stats_difference"] > 0 and row["score_difference"] > 0):
         return "Win"
     elif (row["stats_difference"] < 0 and row["score_difference"] < 0):
         return "Win"
     elif (row["stats_difference"] > 0 and row["score_difference"] < 0):
         return "Loss"
     elif (row["stats_difference"] < 0 and row["score_difference"] > 0):
         return "Loss"
     elif (row["stats_difference"] > 0 and row["score_difference"] == 0):
         return "Draw"
     elif (row["stats_difference"] < 0 and row["score_difference"] == 0):
         return "Draw"

def correct_stats_difference(data):
    galibiyet_indices = data[data["difference_category"] == "Win"].index
    data.loc[galibiyet_indices, "stats_difference"] = data.loc[galibiyet_indices, "stats_difference"].abs()

    maglubiyet_indices = data[data["difference_category"] == "Loss"].index
    data.loc[maglubiyet_indices, "stats_difference"] = -data.loc[maglubiyet_indices, "stats_difference"].abs()

    return data

def create_match_statistics_impact_analysis_plot(merged_data, selected_variable, last_round):
    if any(variable in selected_variable for variable in match_performance_posneg["Positive"]):
        merged_data["stats_difference"] = merged_data["home_team_stat"] - merged_data["away_team_stat"]
    elif any(variable in selected_variable for variable in match_performance_posneg["Negative"]):
        merged_data["stats_difference"] = (merged_data["home_team_stat"] - merged_data["away_team_stat"]) * (-1)

    merged_data["score_difference"] = merged_data["home_score"] - merged_data["away_score"]

    merged_data["difference_category"] = merged_data.apply(classify_difference, axis=1)
    merged_data = correct_stats_difference(merged_data)

    merged_data = merged_data.dropna()

    total_matches = len(merged_data)
    win_count = len(merged_data[merged_data["difference_category"] == "Win"])
    loss_count = len(merged_data[merged_data["difference_category"] == "Loss"])
    draw_count = len(merged_data[merged_data["difference_category"] == "Draw"])

    win_rate = (win_count / total_matches) * 100
    loss_rate = (loss_count / total_matches) * 100
    draw_rate = (draw_count / total_matches) * 100

    categories = ["Win", "Loss", "Draw"]
    rates = [win_rate, loss_rate, draw_rate]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(categories, rates, color=["green", "red", "blue"], edgecolor="black", alpha=0.8)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"%{rate:.1f}",
            ha="center",
            fontsize=10
        )
    ax.set_title(
        f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Match Outcomes When the Team Excelling in the Statistic Wins\n\n{selected_variable}",
        fontsize=12,
        fontweight="bold",
        pad=20
    )
    ax.set_xlabel("Match Outcome", fontsize=12, labelpad=15)
    ax.set_ylabel("Percentage (%)", fontsize=12, labelpad=15)
    ax.set_ylim(0, max(rates) + 10)

    fig.tight_layout()
    ax.grid(True, linestyle="--", alpha=0.3)

    add_footer(fig, y=-0.05)

    st.pyplot(fig)

def main(selected_variable):

    try:

        match_data_df = get_data("match_data")
        match_stats_data_df = get_data("match_stats_data")
        shots_data_df = get_data("shots_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]
        match_data_df = match_data_df[["tournament", "season", "week", "game_id", "home_team", "away_team"]]

        match_stats_data_df = match_stats_data_df[match_stats_data_df["period"] == "ALL"]

        percent_keywords = ["Ball possession", "Tackles won", "Duels"]
        parenthesis_keywords = ["Final third phase", "Long balls", "Crosses", "Ground duels", "Aerial duels", "Dribbles"]
        target_columns = ["home_team_stat", "away_team_stat"]
        match_stats_data_df = clean_percent_columns(match_stats_data_df, percent_keywords, target_columns)
        match_stats_data_df = clean_parenthesis_columns(match_stats_data_df, parenthesis_keywords, target_columns)

        shots_data_df = shots_data_df.merge(
            match_data_df,
            on=["tournament", "season", "week", "game_id"],
            how="left"
        )
        shots_data_df["team_name"] = shots_data_df.apply(
            lambda row: row["home_team"] if row["is_home"] else row["away_team"], axis=1
        )
        shots_data_df["is_goal"] = shots_data_df["shot_type"].apply(lambda x: 1 if x == "goal" else 0)
        shot_maps_data_goal = shots_data_df[shots_data_df["is_goal"] == 1]
        shot_maps_data_goal = shot_maps_data_goal[["tournament", "season", "week", "game_id", "team_name"]]

        match_data_df = match_data_df.copy()
        match_data_df.loc[:, "home_score"] = 0
        match_data_df.loc[:, "away_score"] = 0

        for i, row in match_data_df.iterrows():
            home_team_shots = len(shot_maps_data_goal[(shot_maps_data_goal["game_id"] == row["game_id"]) & (shot_maps_data_goal["team_name"] == row["home_team"])])
            away_team_shots = len(shot_maps_data_goal[(shot_maps_data_goal["game_id"] == row["game_id"]) & (shot_maps_data_goal["team_name"] == row["away_team"])])
            match_data_df.at[i, "home_score"] = home_team_shots
            match_data_df.at[i, "away_score"] = away_team_shots

        merged_data = pd.merge(
            match_data_df,
            match_stats_data_df,
            on=["tournament", "season", "week", "game_id"],
            how="inner"
        )

        if "(Success)" in selected_variable or "(Total)" in selected_variable:
            base_stat = selected_variable.split(" (")[0]
            merged_data = merged_data[merged_data["stat_name"] == base_stat]

            if "(Success)" in selected_variable:
                merged_data["home_team_stat"] = merged_data["home_team_stat"].apply(
                    lambda x: int(x.split("/")[0]) / int(x.split("/")[1]) * 100 if "/" in x else None
                )
                merged_data["away_team_stat"] = merged_data["away_team_stat"].apply(
                    lambda x: int(x.split("/")[0]) / int(x.split("/")[1]) * 100 if "/" in x else None
                )
            elif "(Total)" in selected_variable:
                merged_data["home_team_stat"] = merged_data["home_team_stat"].apply(
                    lambda x: int(x.split("/")[1]) if "/" in x else None
                )
                merged_data["away_team_stat"] = merged_data["away_team_stat"].apply(
                    lambda x: int(x.split("/")[1]) if "/" in x else None
                )
        else:
            merged_data = merged_data[merged_data["stat_name"] == selected_variable]

        merged_data["home_team_stat"] = pd.to_numeric(merged_data["home_team_stat"], errors="coerce").fillna(0)
        merged_data["away_team_stat"] = pd.to_numeric(merged_data["away_team_stat"], errors="coerce").fillna(0)

        last_round = match_data_df["week"].max()

        create_match_statistics_impact_analysis_plot(merged_data, selected_variable, last_round)

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