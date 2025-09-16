import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import require_session_data
from src.analyses.prediction.models.bivariate_poisson_model import (
    solve_bivariate_parameters,
    bivariate_poisson_simulate_match
)
from src.utils.session_data import require_session_data

@st.cache_data(show_spinner=False)
def solve_bivariate_parameters_cached(dataset):
    return solve_bivariate_parameters(dataset)

@st.cache_data(show_spinner=False)
def bivariate_poisson_simulate_match_cached(params_dict, home_team, away_team, max_goals=10):
    return bivariate_poisson_simulate_match(params_dict, home_team, away_team, max_goals)

def run(country: str, league: str, season: str):
    match_df, historical_df = require_session_data("match_data", "tff_historical_matches")

    all_seasons = historical_df['season'].dropna().unique()
    selected_seasons = st.multiselect(
        "Select season(s) to include:",
        options=all_seasons,
        default=list(all_seasons),
        key="bivariate_poisson_seasons"
    )
    if not selected_seasons:
        st.warning("Please select at least one season.")
        return

    filtered_historical = historical_df[historical_df["season"].isin(selected_seasons)]
    match_df_filtered = match_df[match_df['season'] == season]

    if match_df_filtered.empty or match_df_filtered['week'].isna().all():
        st.warning("No matches found for the selected season.")
        return

    last_week = match_df_filtered['week'].max()
    last_week_matches = match_df_filtered[match_df_filtered['week'] == last_week].reset_index(drop=True)
    if last_week_matches.empty:
        st.warning("No matches found for the last week.")
        return

    week_teams = pd.unique(last_week_matches[['home_team', 'away_team']].values.ravel())

    filtered_historical = filtered_historical[
        filtered_historical['home_team'].isin(week_teams) &
        filtered_historical['away_team'].isin(week_teams)
    ]
    filtered_historical = filtered_historical[
        ~(
            (filtered_historical['season'] == season) &
            (filtered_historical['week'] >= last_week)
        )
    ]

    filtered_historical["home_score"] = pd.to_numeric(filtered_historical["home_score"], errors="coerce")
    filtered_historical["away_score"] = pd.to_numeric(filtered_historical["away_score"], errors="coerce")
    filtered_historical = filtered_historical.dropna(subset=["home_score", "away_score"])
    filtered_historical = filtered_historical.rename(
        columns={
            "home_score": "home_team_goals",
            "away_score": "away_team_goals"
        }
    )

    if filtered_historical.empty:
        st.warning("No historical data available after filtering.")
        return

    filtered_historical["home_team_goals"] = filtered_historical["home_team_goals"].astype(int)
    filtered_historical["away_team_goals"] = filtered_historical["away_team_goals"].astype(int)

    if st.button("Run Model for All Matches", key="bivariate_poisson_calc_button_all"):
        with st.spinner("Running the model for all matches... It might take a bit longer while the system is being set up."):
            params = solve_bivariate_parameters_cached(filtered_historical)

        st.markdown(f"### {season} {league} - Week {last_week} Predictions (Bivariate Poisson)")

        for _, match_row in last_week_matches.iterrows():
            home_team = match_row["home_team"]
            away_team = match_row["away_team"]

            model_df = bivariate_poisson_simulate_match_cached(params, home_team, away_team, max_goals=10)

            if model_df is None or not hasattr(model_df, "shape") or model_df.size == 0 or not np.any(model_df):
                st.warning(f"No matrix to plot for {home_team} - {away_team}.")
                continue

            home_win_prob = np.sum(np.tril(model_df, -1)) * 100.0
            draw_prob = np.sum(np.diag(model_df)) * 100.0
            away_win_prob = np.sum(np.triu(model_df, 1)) * 100.0

            percentage_matrix = model_df * 100.0

            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(
                percentage_matrix,
                annot=True,
                fmt=".1f",
                cmap="Reds",
                cbar=False,
                ax=ax
            )
            ax.set_xlabel(f"{away_team} (Away) Goals", fontsize=12, labelpad=10)
            ax.set_ylabel(f"{home_team} (Home) Goals", fontsize=12)
            ax.set_title(
                f"{season} {league} - Scoreline Outcome Probabilities for Week {last_week}\n"
                "Predicted by Bivariate Poisson",
                fontsize=12,
                fontweight="bold",
                pad=25
            )
            ax.text(
                0.5, 1.115,
                f"{home_team} Win: {home_win_prob:.1f}% | Draw: {draw_prob:.1f}% | {away_team} Win: {away_win_prob:.1f}%",
                fontsize=11,
                color="dimgray",
                ha='center',
                va='bottom',
                transform=ax.transAxes
            )
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            plt.tight_layout()
            st.subheader(f"{home_team} - {away_team}")
            st.pyplot(fig)
            plt.close(fig)