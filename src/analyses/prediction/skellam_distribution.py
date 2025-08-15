import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import skellam
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import require_session_data

def solve_skellam_parameters(dataset, init_vals=None, options={"disp": False, "maxiter": 100}, **kwargs):
    teams = np.sort(list(set(dataset["home_team"].unique()) | set(dataset["away_team"].unique())))
    n_teams = len(teams)

    if "home_team_goals" not in dataset.columns or "away_team_goals" not in dataset.columns:
        tmp = dataset.rename(columns={"home_score":"home_team_goals", "away_score":"away_team_goals"})
    else:
        tmp = dataset

    if init_vals is None:
        gf_home = tmp.groupby("home_team")["home_team_goals"].mean()
        gf_away = tmp.groupby("away_team")["away_team_goals"].mean()
        gf = pd.concat([gf_home.rename("gf_h"), gf_away.rename("gf_a")], axis=1).mean(axis=1).reindex(teams).fillna(1.0).values

        ga_home = tmp.groupby("home_team")["away_team_goals"].mean()
        ga_away = tmp.groupby("away_team")["home_team_goals"].mean()
        ga = pd.concat([ga_home.rename("ga_h"), ga_away.rename("ga_a")], axis=1).mean(axis=1).reindex(teams).fillna(1.0).values

        avg_attack  = np.log(np.maximum(gf, 1e-6))
        avg_defence = -np.log(np.maximum(ga, 1e-6))

        avg_attack  = avg_attack  - avg_attack.mean()
        avg_defence = avg_defence - avg_defence.mean()

        init_vals = np.concatenate([avg_attack, avg_defence, np.array([0.0])])

    def estimate_params(params):
        attack_coeffs = dict(zip(teams, params[:n_teams]))
        defence_coeffs = dict(zip(teams, params[n_teams:2*n_teams]))
        home_adv = params[-1]

        ll = 0.0
        for row in dataset.itertuples():
            gd = getattr(row, "home_team_goals") - getattr(row, "away_team_goals")

            mu1 = np.exp(attack_coeffs[getattr(row, "home_team")] + defence_coeffs[getattr(row, "away_team")] + home_adv)
            mu2 = np.exp(attack_coeffs[getattr(row, "away_team")] + defence_coeffs[getattr(row, "home_team")])

            p = skellam.pmf(gd, mu1, mu2)
            ll += np.log(max(p, 1e-12))
        return -ll

    constraints = [
        {"type": "eq", "fun": lambda x, n=n_teams: np.sum(x[:n])},
        {"type": "eq", "fun": lambda x, n=n_teams: np.sum(x[n:2*n])}
    ]

    bounds = Bounds(
        [-np.inf] * n_teams + [-np.inf] * n_teams + [-2.0],
        [ np.inf] * n_teams + [ np.inf] * n_teams + [ 2.0]
    )

    opt_output = minimize(
        estimate_params,
        init_vals,
        method="SLSQP",
        options=options,
        constraints=constraints,
        bounds=bounds,
        **kwargs
    )

    return dict(
        zip(
            ["attack_" + team for team in teams] +
            ["defence_" + team for team in teams] +
            ["home_adv"],
            opt_output.x
        )
    )

def skellam_goal_diff_probs(params_dict, home_team, away_team, max_goals=10):
    mu1 = np.exp(params_dict["attack_" + home_team] + params_dict["defence_" + away_team] + params_dict["home_adv"])
    mu2 = np.exp(params_dict["attack_" + away_team] + params_dict["defence_" + home_team])
    diffs = np.arange(-max_goals, max_goals + 1)
    diff_probs = skellam.pmf(diffs, mu1, mu2)
    diff_probs /= diff_probs.sum()
    return diffs, diff_probs

@st.cache_data(show_spinner=False)
def solve_skellam_parameters_cached(dataset):
    return solve_skellam_parameters(dataset)

@st.cache_data(show_spinner=False)
def skellam_goal_diff_probs_cached(params_dict, home_team, away_team, max_goals=10):
    return skellam_goal_diff_probs(params_dict, home_team, away_team, max_goals)

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
    match_df_filtered = match_df[match_df['season'] == season]

    last_week = match_df_filtered['week'].max()
    last_week_matches = match_df_filtered[match_df_filtered['week'] == last_week].reset_index(drop=True)

    option_map = {
        f"{row.home_team} - {row.away_team}": i
        for i, row in last_week_matches.iterrows()
    }
    match_options_unique = list(option_map.keys())

    selected_match = st.selectbox(
        f"Select a match from week {last_week} ({len(match_options_unique)} matches):",
        match_options_unique,
        index=None,
        placeholder="Select a match",
        key="skellam_selected_match"
    )

    if selected_match:
        row_idx = option_map[selected_match]
        match_row = last_week_matches.iloc[row_idx]
        home_team = match_row["home_team"]
        away_team = match_row["away_team"]

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
        filtered_historical = filtered_historical.dropna(subset=["home_score", "away_score"]).copy()
        filtered_historical = filtered_historical.rename(
            columns={"home_score": "home_team_goals", "away_score": "away_team_goals"}
        )

        if st.button("Run Model", key="skellam_calc_button"):
            with st.spinner("Running the model... The first run might take a bit longer while the system is being set up."):
                params = solve_skellam_parameters_cached(filtered_historical)
                diffs, diff_probs = skellam_goal_diff_probs_cached(params, home_team, away_team, max_goals=10)

            if diffs is not None and diff_probs is not None and np.any(diff_probs):
                home_win = diff_probs[diffs > 0].sum() * 100
                draw = diff_probs[diffs == 0].sum() * 100
                away_win = diff_probs[diffs < 0].sum() * 100

                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#e74c3c' if d < 0 else '#f39c12' if d == 0 else '#2471a3' for d in diffs]
                bars = ax.bar(diffs, diff_probs * 100, color=colors)

                for i, v in enumerate(diff_probs * 100):
                    if v > 0.5:
                        ax.text(diffs[i], v + 0.3, f"{v:.1f}%", ha='center', va='bottom', fontsize=6, fontweight='bold')

                ax.set_xticks(diffs)
                ax.set_xticklabels(diffs)
                plt.setp(ax.get_xticklabels(), fontsize=10)

                ax.set_xlabel(f"{home_team} - {away_team} (Score Difference)", fontsize=14, labelpad=20)
                ax.set_ylabel("Probability (%)", fontsize=14, labelpad=20)
                ax.set_title(
                    f"{season} {league} - Scoreline Outcome Probabilities for Week {last_week}\n"
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
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No score difference distribution to plot.")