import streamlit as st
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_player_shot_location_plot(df_goals, df_non_goals, team, player_name):
    pitch = VerticalPitch(
        pitch_type="opta",
        corner_arcs=True,
        half=False,
        label=False,
        tick=False
    )

    fig, ax = pitch.draw(figsize=(16, 16))

    pitch.scatter(
        df_non_goals["player_coordinates_x"],
        df_non_goals["player_coordinates_y"],
        edgecolors="black",
        c="gray",
        marker="h",
        alpha=0.3,
        s=150,
        label="Not a Goal",
        ax=ax
    )

    pitch.scatter(
        df_goals["player_coordinates_x"],
        df_goals["player_coordinates_y"],
        edgecolors="black",
        c="red",
        marker="h",
        alpha=0.7,
        s=150,
        label="Goal",
        ax=ax
    )

    total_shots = len(df_goals) + len(df_non_goals)
    conversion_rate = len(df_goals) / total_shots * 100 if total_shots > 0 else 0

    non_penalty_goals = df_goals[df_goals["situation"] != "penalty"]
    non_penalty_shots = total_shots - len(df_goals[df_goals["situation"] == "penalty"])
    non_penalty_conversion_rate = len(non_penalty_goals) / non_penalty_shots * 100 if non_penalty_shots > 0 else 0

    ax.set_title(
        f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Player Shot Locations",
        fontsize=16,
        fontweight="bold",
        y=1.05
    )
    ax.text(
        0.5, 1.04,
        f"{player_name} ({team})",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        transform=ax.transAxes
    )

    add_footer(fig, x=0.5, y=0, ha="center")

    ax.text(
        0.5, 1.02,
        s=(
            f"Shot Conversion Rate: %{conversion_rate:.1f} (Including Penalties), %{non_penalty_conversion_rate:.1f} (Excluding Penalties)"
        ),
        size=12,
        va="center",
        ha="center",
        transform=ax.transAxes
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=12,
        frameon=False,
        facecolor="white",
        edgecolor="black"
    )

    st.pyplot(fig)

def main(team, player):
    try:

        match_data_df = get_data("match_data")
        shots_data_df = get_data("shots_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]
        match_data_df = match_data_df[["season", "week", "game_id", "home_team", "away_team"]]

        shots_data_df = shots_data_df[[
            "season", "week", "game_id", "player_name", "is_home", "shot_type", "situation",
            "goal_mouth_location", "player_coordinates_x", "player_coordinates_y"
        ]]

        shots_data_df = shots_data_df.merge(
            match_data_df,
            on=["season", "week", "game_id"],
            how="left"
        )

        shots_data_df["team_name"] = shots_data_df.apply(
            lambda row: row["home_team"] if row["is_home"] else row["away_team"], axis=1
        )

        shots_data_df["is_goal"] = shots_data_df["shot_type"].apply(lambda x: 1 if x == "goal" else 0)
        shots_data_df["player_coordinates_x"] = 100 - shots_data_df["player_coordinates_x"]
        shots_data_df["player_coordinates_y"] = 100 - shots_data_df["player_coordinates_y"]

        player_data = shots_data_df[(shots_data_df["team_name"] == team) & (shots_data_df["player_name"] == player)]

        df_goals = player_data[player_data["is_goal"] == 1]
        df_non_goals = player_data[player_data["is_goal"] == 0]

        create_player_shot_location_plot(df_goals, df_non_goals, team, player)

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