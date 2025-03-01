import streamlit as st
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_rating_plot(rating_data, last_round, subcategory):

    if subcategory in ["Mean-Standard Deviation (Overall)", "Mean-Standard Deviation (Home)", "Mean-Standard Deviation (Away)"]:

        context = subcategory[subcategory.find('(')+1:subcategory.find(')')]

        fig, ax = plt.subplots(figsize=(12, 10))

        ax.scatter(
            rating_data["mean"],
            rating_data["std"],
            alpha=0
        )

        mean_avg_rating_data = rating_data["mean"].mean()
        mean_sd_rating_data = rating_data["std"].mean()

        ax.axvline(x=mean_avg_rating_data, color="darkblue", linestyle="--", linewidth=2, label="League-Wide Average Rating")
        ax.axhline(y=mean_sd_rating_data, color="darkred", linestyle="--", linewidth=2, label="League-Wide Standard Deviation")

        def getImage(path):
            return OffsetImage(plt.imread(path), zoom=.3, alpha=1)

        for index, row in rating_data.iterrows():
            logo_path = f"./imgs/{st.session_state['selected_league']}_logos/{row['team_name']}.png"
            ab = AnnotationBbox(getImage(logo_path), (row["mean"], row["std"]), frameon=False)
            ax.add_artist(ab)

        ax.set_xlabel("Mean", labelpad=20, fontsize=12)
        ax.set_ylabel("Standard Deviation", labelpad=20, fontsize=12)
        ax.set_title(
            f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Teams' {context} Average Rating and Performance Consistency",
            fontsize=14,
            fontweight="bold",
            pad=40
        )
        ax.grid(True, linestyle="--", alpha=0.7)

        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            fontsize=10,
            frameon=False
        )

        add_footer(fig)
        st.pyplot(fig)

def main(subcategory):
    try:

        match_data_df = get_data("match_data")
        lineups_data_df = get_data("lineups_data")

        lineups_data_df = lineups_data_df[lineups_data_df["stat_name"] == "rating"]

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]

        lineups_games_data = lineups_data_df.merge(
            match_data_df,
            on=["tournament","season","week","game_id"],
            how="right"
        )

        lineups_games_data["team_name"] = lineups_games_data.apply(
            lambda row: row["home_team"] if row["team"] == "home" else row["away_team"], axis=1
        )

        last_round = match_data_df["week"].max()

        if subcategory == "Mean-Standard Deviation (Overall)":
            rating_data = lineups_games_data.groupby("team_name")["stat_value"].agg(["mean", "std"]).reset_index()
            create_rating_plot(rating_data, last_round, subcategory)
        elif subcategory == "Mean-Standard Deviation (Home)":
            home_data = lineups_games_data[lineups_games_data["team"] == "home"]
            rating_data = home_data.groupby("team_name")["stat_value"].agg(["mean", "std"]).reset_index()
            create_rating_plot(rating_data, last_round, subcategory)
        elif subcategory == "Mean-Standard Deviation (Away)":
            away_data = lineups_games_data[lineups_games_data["team"] == "away"]
            rating_data = away_data.groupby("team_name")["stat_value"].agg(["mean", "std"]).reset_index()
            create_rating_plot(rating_data, last_round, subcategory)

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