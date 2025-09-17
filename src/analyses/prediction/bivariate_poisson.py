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

def _plot_bp_match(params, home_team, away_team, season, league, week):
    model_df = bivariate_poisson_simulate_match_cached(params, home_team, away_team, max_goals=10)

    if model_df is None or not hasattr(model_df, "shape") or model_df.size == 0 or not np.any(model_df):
        st.warning(f"No matrix to plot for {home_team} vs. {away_team}.")
        return

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
        f"{season} {league} - Week {week} Scoreline Outcome Probabilities\n"
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
    st.subheader(f"{home_team} vs. {away_team}")
    st.pyplot(fig)
    plt.close(fig)

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

    season_df = match_df[match_df['season'] == season].copy()
    if season_df.empty or season_df['week'].isna().all():
        st.warning("No matches found for the selected season.")
        return

    include_postponed = st.checkbox(
        "Include postponed matches",
        value=False,
        help="Turn on to also show postponed matches from earlier weeks. They will appear under their original weeks.",
        key="include_postponed_bp"
    )

    status_norm = season_df["status"].fillna("").str.strip().str.lower()

    last_week = season_df["week"].max()

    postponed_df = season_df[status_norm == "postponed"].copy() if include_postponed else season_df.iloc[0:0].copy()
    upcoming_df = season_df[(season_df["week"] == last_week) & (status_norm != "postponed")].copy()

    target_matches = pd.concat([postponed_df, upcoming_df], ignore_index=True)
    if target_matches.empty:
        st.warning("No matches found to display for the current settings.")
        return

    week_teams = pd.unique(target_matches[['home_team', 'away_team']].values.ravel())
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
    filtered_historical = filtered_historical.dropna(subset=["home_score", "away_score"]).rename(
        columns={"home_score": "home_team_goals", "away_score": "away_team_goals"}
    )
    if filtered_historical.empty:
        st.warning("No historical data available after filtering.")
        return

    filtered_historical["home_team_goals"] = filtered_historical["home_team_goals"].astype(int)
    filtered_historical["away_team_goals"] = filtered_historical["away_team_goals"].astype(int)

    if st.button("Run Model for All Matches", key="bivariate_poisson_calc_button_all"):
        with st.spinner("Running the model for all matches... It might take a bit longer while the system is being set up."):
            params = solve_bivariate_parameters_cached(filtered_historical)

        if include_postponed and not postponed_df.empty:
            st.markdown("## Postponed Matches")
            weeks_postponed = sorted([int(w) for w in postponed_df["week"].dropna().unique()])
            for wk in weeks_postponed:
                wk_group = postponed_df[postponed_df["week"] == wk]
                st.markdown(f"### {season} {league} - Week {wk} Predictions (Postponed) (Bivariate Poisson)")
                for _, match_row in wk_group.iterrows():
                    _plot_bp_match(
                        params=params,
                        home_team=match_row["home_team"],
                        away_team=match_row["away_team"],
                        season=season,
                        league=league,
                        week=int(wk)
                    )

        if not upcoming_df.empty:
            st.markdown("## Upcoming Week")
            st.markdown(f"### {season} {league} - Week {int(last_week)} Predictions (Bivariate Poisson)")
            for _, match_row in upcoming_df.iterrows():
                _plot_bp_match(
                    params=params,
                    home_team=match_row["home_team"],
                    away_team=match_row["away_team"],
                    season=season,
                    league=league,
                    week=int(last_week)
                )