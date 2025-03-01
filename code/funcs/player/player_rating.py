import streamlit as st
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_player_rating_plot(main_rating_df, team, player):

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.fill_between(
        main_rating_df["week"],
        main_rating_df["min"],
        main_rating_df["max"],
        color="gray",
        alpha=0.1,
        label="Team Min-Max Rating Channel"
    )
    ax.plot(
        main_rating_df["week"],
        main_rating_df["min"],
        color="gray",
        alpha=0.2,
        linewidth=2,
        linestyle="-"
    )
    ax.plot(
        main_rating_df["week"],
        main_rating_df["max"],
        color="gray",
        alpha=0.2,
        linewidth=2,
        linestyle="-"
    )
    ax.plot(
        main_rating_df["week"],
        main_rating_df["mean"],
        label="Team Average Rating",
        marker="o",
        linestyle="--",
        linewidth=2,
        color="gray"
    )
    ax.plot(
        main_rating_df["week"],
        main_rating_df["stat_value"],
        label="Player Rating",
        marker="o",
        linestyle="-",
        color="red"
    )

    ax.set_xlabel("Week", labelpad=20, fontsize=12)
    ax.set_ylabel("Rating", labelpad=20, fontsize=12)
    ax.set_title(
        f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Player Ratings\n\n{player} ({team})",
        fontsize=14,
        fontweight="bold",
        pad=40
    )

    ax.set_ylim(0, 10)
    ax.set_xticks(main_rating_df["week"].astype(int))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.xaxis.set_major_locator(MultipleLocator(3))

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        fontsize=8,
        frameon=False
    )

    ax.grid(True, linestyle="--", alpha=0.7)

    add_footer(fig)

    st.pyplot(fig)

def main(team, player):
    try:

        match_data_df = get_data("match_data")
        lineups_data_df = get_data("lineups_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]
        match_data_df = match_data_df[["tournament","season","week","game_id","home_team","away_team"]]

        player_rating_df = match_data_df.merge(
            lineups_data_df,
            on=["tournament","season","week","game_id"]
        )

        player_rating_df = player_rating_df[player_rating_df["stat_name"] == "rating"]
        player_rating_df["team_name"] = player_rating_df.apply(
            lambda row: row["home_team"] if row["team"] == "home" else row["away_team"], axis=1
        )
        player_rating_df = player_rating_df[player_rating_df['team_name'] == team]
        rating_df_filtered_player = player_rating_df[player_rating_df["player_name"] == player]

        team_min_max_rating_df = player_rating_df.groupby(["team_name", "week"])["stat_value"].agg(["min", "max", "mean"]).reset_index()

        main_rating_df = team_min_max_rating_df.merge(
            rating_df_filtered_player,
            on=["team_name","week"],
            how="left"
        )

        create_player_rating_plot(main_rating_df, team, player)

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