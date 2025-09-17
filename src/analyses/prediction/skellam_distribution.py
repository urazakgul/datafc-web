import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import require_session_data
from src.analyses.prediction.models.skellam_distribution_model import (
    solve_skellam_parameters,
    skellam_goal_diff_probs
)
from src.utils.session_data import require_session_data

@st.cache_data(show_spinner=False)
def solve_skellam_parameters_cached(dataset):
    return solve_skellam_parameters(dataset)

@st.cache_data(show_spinner=False)
def skellam_goal_diff_probs_cached(params_dict, home_team, away_team, max_goals=10):
    return skellam_goal_diff_probs(params_dict, home_team, away_team, max_goals)

def _plot_skellam_match(params, home_team, away_team, season, league, week):
    diffs, diff_probs = skellam_goal_diff_probs_cached(params, home_team, away_team, max_goals=10)

    if diffs is None or diff_probs is None or not np.any(diff_probs):
        st.warning(f"No score difference distribution to plot for {home_team} vs. {away_team}.")
        return

    home_win = float(diff_probs[diffs > 0].sum() * 100.0)
    draw = float(diff_probs[diffs == 0].sum() * 100.0)
    away_win = float(diff_probs[diffs < 0].sum() * 100.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c' if d < 0 else '#f39c12' if d == 0 else '#2471a3' for d in diffs]
    bars = ax.bar(diffs, diff_probs * 100.0, color=colors)

    for i, v in enumerate(diff_probs * 100.0):
        if v > 0.5:
            ax.text(diffs[i], v + 0.3, f"{v:.1f}%", ha='center', va='bottom', fontsize=6, fontweight='bold')

    ax.set_xticks(diffs)
    ax.set_xticklabels(diffs)
    plt.setp(ax.get_xticklabels(), fontsize=10)

    ax.set_xlabel(f"{home_team} vs. {away_team} (Score Difference)", fontsize=14, labelpad=20)
    ax.set_ylabel("Probability (%)", fontsize=14, labelpad=20)
    ax.set_title(
        f"{season} {league} - Week {week} Scoreline Outcome Probabilities\n"
        "Predicted by Skellam Distribution",
        fontsize=12,
        fontweight="bold",
        pad=25
    )
    ax.text(
        0.5, 1.03,
        f"{home_team} Win: {home_win:.1f}% | Draw: {draw:.1f}% | {away_team} Win: {away_win:.1f}%",
        fontsize=11,
        color="dimgray",
        ha='center',
        va='bottom',
        transform=ax.transAxes
    )
    ax.grid(True, linestyle="--", alpha=0.7)
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
        key="skellam_seasons"
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
        key="include_postponed_skellam"
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
    filtered_historical = filtered_historical.dropna(subset=["home_score", "away_score"]).copy()
    filtered_historical = filtered_historical.rename(
        columns={"home_score": "home_team_goals", "away_score": "away_team_goals"}
    )

    if filtered_historical.empty:
        st.warning("No historical data available after filtering.")
        return

    if not pd.api.types.is_integer_dtype(filtered_historical["home_team_goals"]):
        filtered_historical["home_team_goals"] = filtered_historical["home_team_goals"].astype(int)
    if not pd.api.types.is_integer_dtype(filtered_historical["away_team_goals"]):
        filtered_historical["away_team_goals"] = filtered_historical["away_team_goals"].astype(int)

    if st.button("Run Model for All Matches", key="skellam_calc_button_all"):
        with st.spinner("Running the model for all matches... It might take a bit longer while the system is being set up."):
            params = solve_skellam_parameters_cached(filtered_historical)

        if include_postponed and not postponed_df.empty:
            st.markdown("## Postponed Matches")
            weeks_postponed = sorted([int(w) for w in postponed_df["week"].dropna().unique()])
            for wk in weeks_postponed:
                wk_group = postponed_df[postponed_df["week"] == wk]
                st.markdown(f"### {season} {league} - Week {wk} Predictions (Postponed) (Skellam Distribution)")
                for _, match_row in wk_group.iterrows():
                    _plot_skellam_match(
                        params=params,
                        home_team=match_row["home_team"],
                        away_team=match_row["away_team"],
                        season=season,
                        league=league,
                        week=int(wk)
                    )

        if not upcoming_df.empty:
            st.markdown("## Upcoming Week")
            st.markdown(f"### {season} {league} - Week {int(last_week)} Predictions (Skellam Distribution)")
            for _, match_row in upcoming_df.iterrows():
                _plot_skellam_match(
                    params=params,
                    home_team=match_row["home_team"],
                    away_team=match_row["away_team"],
                    season=season,
                    league=league,
                    week=int(last_week)
                )