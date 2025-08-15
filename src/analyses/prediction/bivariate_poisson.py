import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize, Bounds
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import require_session_data

def bivariate_poisson_pmf(x, y, lambda1, lambda2, lambda3):
    min_k = int(min(x, y))
    prob = 0.0
    for k in range(min_k + 1):
        prob += (
            poisson.pmf(x - k, lambda1)
            * poisson.pmf(y - k, lambda2)
            * poisson.pmf(k, lambda3)
        )
    return prob

def bivariate_log_like(x, y, alpha_x, beta_x, alpha_y, beta_y, gamma, log_lambda3):
    l1 = np.exp(alpha_x + beta_y + gamma)
    l2 = np.exp(alpha_y + beta_x)
    l3 = np.exp(log_lambda3)
    pmf = bivariate_poisson_pmf(x, y, l1, l2, l3)
    return np.log(max(pmf, 1e-12))

def solve_bivariate_parameters(
    dataset: pd.DataFrame,
    init_vals=None,
    options={"disp": False, "maxiter": 150},
    **kwargs
):
    teams = np.sort(
        list(set(dataset["home_team"].unique()) | set(dataset["away_team"].unique()))
    )
    n_teams = len(teams)

    if init_vals is None:
        mu_home = dataset["home_team_goals"].mean()
        mu_away = dataset["away_team_goals"].mean()

        team_home_means = (
            dataset.groupby("home_team")["home_team_goals"]
            .mean()
            .reindex(teams)
            .fillna(mu_home)
            .values
        )
        team_away_means = (
            dataset.groupby("away_team")["away_team_goals"]
            .mean()
            .reindex(teams)
            .fillna(mu_away)
            .values
        )

        init_attack = np.log(np.clip(team_home_means, 1e-6, None)) - np.log(max(mu_home, 1e-6))
        init_defence = -(
            np.log(np.clip(team_away_means, 1e-6, None)) - np.log(max(mu_away, 1e-6))
        )

        init_attack -= init_attack.mean()
        init_defence -= init_defence.mean()

        init_vals = np.concatenate([init_attack, init_defence, np.array([0.0, np.log(0.05)])])

    def estimate_params(params):
        attack_coeffs = dict(zip(teams, params[:n_teams]))
        defence_coeffs = dict(zip(teams, params[n_teams:2 * n_teams]))
        gamma = params[-2]
        log_lambda3 = params[-1]

        ll = 0.0
        for row in dataset.itertuples():
            ll += bivariate_log_like(
                row.home_team_goals,
                row.away_team_goals,
                attack_coeffs[row.home_team],
                defence_coeffs[row.away_team],
                attack_coeffs[row.away_team],
                defence_coeffs[row.home_team],
                gamma,
                log_lambda3,
            )
        return -ll

    constraints = [
        {"type": "eq", "fun": lambda p, n=n_teams: np.sum(p[:n])},
        {"type": "eq", "fun": lambda p, n=n_teams: np.sum(p[n:2*n])},
    ]

    bounds = Bounds(
        [-np.inf] * n_teams + [-np.inf] * n_teams + [-3, -4],
        [ np.inf] * n_teams + [ np.inf] * n_teams + [ 3,  4],
    )

    opt_output = minimize(
        estimate_params,
        init_vals,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=options,
        **kwargs
    )

    return dict(
        zip(
            [f"attack_{team}" for team in teams] +
            [f"defence_{team}" for team in teams] +
            ["home_adv", "log_lambda3"],
            opt_output.x
        )
    )

def bivariate_poisson_simulate_match(params_dict, home_team, away_team, max_goals=10):
    l1 = np.exp(params_dict[f"attack_{home_team}"] + params_dict[f"defence_{away_team}"] + params_dict["home_adv"])
    l2 = np.exp(params_dict[f"attack_{away_team}"] + params_dict[f"defence_{home_team}"])
    l3 = np.exp(params_dict["log_lambda3"])

    output_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            output_matrix[x, y] = bivariate_poisson_pmf(x, y, l1, l2, l3)

    total_mass = output_matrix.sum()
    if total_mass > 0:
        output_matrix /= total_mass

    return output_matrix

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
        key="bivariate_poisson_selected_match"
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
        filtered_historical = filtered_historical.dropna(subset=["home_score", "away_score"])
        filtered_historical = filtered_historical.rename(
            columns={
                "home_score": "home_team_goals",
                "away_score": "away_team_goals"
            }
        )

        filtered_historical["home_team_goals"] = filtered_historical["home_team_goals"].astype(int)
        filtered_historical["away_team_goals"] = filtered_historical["away_team_goals"].astype(int)

        if st.button("Run Model", key="bivariate_poisson_calc_button"):
            with st.spinner("Running the model... The first run might take a bit longer while the system is being set up."):
                params = solve_bivariate_parameters_cached(filtered_historical)
                model_df = bivariate_poisson_simulate_match_cached(params, home_team, away_team, max_goals=10)

            if model_df is not None and hasattr(model_df, "shape") and model_df.size > 0 and np.any(model_df):

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
                st.pyplot(fig)
                plt.close(fig)

            else:
                st.warning("No matrix to plot.")