import streamlit as st
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import event_colors, PLOT_STYLE

plt.style.use(PLOT_STYLE)

def fill_team_name(df):
    df["team_name"] = df.groupby("id")["team_name"].transform(lambda x: x.ffill().bfill())
    return df

def merge_match_data(match_data_df, shots_data_df):
    filtered_shots = shots_data_df[shots_data_df["shot_type"] == "goal"][
        ["tournament", "season", "week", "game_id", "player_name", "is_home", "goal_type", "xg"]
    ]
    merged_df = match_data_df.merge(filtered_shots, on=["tournament", "season", "week", "game_id"])
    return merged_df[~merged_df["goal_type"].isin(["penalty", "own"])]

def create_goal_network_plot(side_data, team):

    player_event_counts = side_data.groupby(["player_name", "event_type"]).size().reset_index(name="count")
    player_total_counts = player_event_counts.groupby("player_name")["count"].sum().reset_index(name="total_count")

    player_event_counts = player_event_counts.merge(player_total_counts, on="player_name")
    player_event_counts["percentage"] = (player_event_counts["count"] / player_event_counts["total_count"]) * 100

    pivot_df = player_event_counts.pivot(index="player_name", columns="event_type", values="count").fillna(0)
    pivot_df["total_count"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values(by="total_count")

    event_columns = [col for col in pivot_df.columns if col != "total_count"]
    colors = [event_colors[event] for event in event_columns if event in event_colors]

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

    ax.set_xlabel("Number of Involvements in Goal Sequences", fontsize=14, labelpad=20)
    ax.set_ylabel("")
    ax.set_title(
        f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – {team} Players' Goal Sequence Involvement by Event Type",
        fontsize=18,
        fontweight="bold",
        pad=70
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=7,
        fontsize=12,
        frameon=False
    )

    ax.grid(True, linestyle="--", alpha=0.7)

    add_footer(fig, x=0.95, y=0.02, fontsize=14)

    st.pyplot(fig)

def main(team=None):
    try:

        match_data_df = get_data("match_data")
        shots_data_df = get_data("shots_data")
        goal_networks_data_df = get_data("goal_networks_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]
        match_data_df = match_data_df[["tournament", "season", "week", "game_id", "home_team", "away_team"]]
        match_shots_data_df = merge_match_data(match_data_df, shots_data_df)

        goal_networks_data_scored_df = goal_networks_data_df.copy()

        goal_networks_data_scored_df["team_name"] = None

        for game_id in match_shots_data_df["game_id"].unique():
            match_data = match_shots_data_df[match_shots_data_df["game_id"] == game_id]
            for _, row in match_data.iterrows():
                team_name = row["home_team"] if row["is_home"] else row["away_team"]

                goal_networks_data_scored_df.loc[
                    (goal_networks_data_scored_df["game_id"] == game_id) &
                    (goal_networks_data_scored_df["player_name"] == row["player_name"]) &
                    (goal_networks_data_scored_df["event_type"] == "goal"), "team_name"
                ] = team_name

        goal_networks_data_scored_df = fill_team_name(goal_networks_data_scored_df)

        for _, group in goal_networks_data_scored_df.groupby("id"):
            if (group["event_type"] == "goal").any() and group.loc[group["event_type"] == "goal", "goal_shot_x"].iloc[0] != 100:
                goal_networks_data_scored_df.loc[group.index, ["player_x", "player_y"]] = 100 - group[["player_x", "player_y"]]

        goal_networks_data_scored_df = goal_networks_data_scored_df.merge(match_data_df, on=["tournament", "season", "week", "game_id"])

        side_data = goal_networks_data_scored_df[goal_networks_data_scored_df["team_name"] == team]

        create_goal_network_plot(side_data, team)

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