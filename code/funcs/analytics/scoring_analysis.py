import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_summary_season_plot(summary_df, selected_scoring_analysis):

    if selected_scoring_analysis == "Home vs Away Goal Matrix":
        fig, ax = plt.subplots(figsize=(12, 12))

        sns.heatmap(summary_df, annot=True, cmap="Reds", fmt=".1%", linewidths=0.5, cbar=False, ax=ax)

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        ax.set_title(
            f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Probability Matrix of Home and Away Team Scores",
            fontsize=16,
            fontweight="bold",
            pad=70
        )
        ax.set_xlabel("Away Score", fontsize=12, labelpad=20)
        ax.set_ylabel("Home Score", fontsize=12, labelpad=20)

        add_footer(fig, x=0.95, y=0.01)

    st.pyplot(fig)

def main(selected_scoring_analysis):
    try:

        match_data_df = get_data("match_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]

        if selected_scoring_analysis == "Home vs Away Goal Matrix":
            max_home = match_data_df["home_score_display"].max()
            max_away = match_data_df["away_score_display"].max()
            max_score = max(max_home, max_away)

            home_scores = match_data_df["home_score_display"]
            away_scores = match_data_df["away_score_display"]

            score_matrix = pd.crosstab(home_scores, away_scores)
            score_matrix = score_matrix.reindex(index=range(max_score + 1), columns=range(max_score + 1), fill_value=0)
            score_percentage_matrix = score_matrix / score_matrix.sum().sum()
            summary_df = score_percentage_matrix.copy()

        create_summary_season_plot(summary_df, selected_scoring_analysis)

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