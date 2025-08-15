import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize, Bounds
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import require_session_data

def rho_correction(x, y, lambda_x, mu_y, rho):
    if x == 0 and y == 0:
        return max(1 - lambda_x * mu_y * rho, 1e-10)
    elif x == 0 and y == 1:
        return 1 + lambda_x * rho
    elif x == 1 and y == 0:
        return 1 + mu_y * rho
    elif x == 1 and y == 1:
        return max(1 - rho, 1e-10)
    else:
        return 1.0

def dc_log_like(x, y, alpha_x, beta_x, alpha_y, beta_y, rho, gamma):
    lambda_x = np.exp(alpha_x + beta_y + gamma)
    mu_y = np.exp(alpha_y + beta_x)
    log_lambda_x = np.log(max(poisson.pmf(x, lambda_x), 1e-10))
    log_mu_y = np.log(max(poisson.pmf(y, mu_y), 1e-10))
    return (
        np.log(max(rho_correction(x, y, lambda_x, mu_y, rho), 1e-10)) + log_lambda_x + log_mu_y
    )

def solve_parameters(dataset, init_vals=None, options={"disp": False, "maxiter": 200}, weights=None, **kwargs):
    teams = np.sort(list(set(dataset["home_team"].unique()) | set(dataset["away_team"].unique())))
    n_teams = len(teams)

    if init_vals is None:
        league_home_mean = dataset["home_team_goals"].mean()
        league_away_mean = dataset["away_team_goals"].mean()

        avg_attack = (
            dataset.groupby("home_team")["home_team_goals"]
            .mean()
            .reindex(teams)
            .fillna(league_home_mean)
            .values
        )
        avg_defence = -(
            dataset.groupby("away_team")["away_team_goals"]
            .mean()
            .reindex(teams)
            .fillna(league_away_mean)
            .values
        )

        avg_attack = avg_attack - np.mean(avg_attack)
        avg_defence = avg_defence - np.mean(avg_defence)

        init_vals = np.concatenate([avg_attack, avg_defence, np.array([0.0, 0.0])])

    def estimate_parameters(params):
        attack_coeffs = dict(zip(teams, params[:n_teams]))
        defence_coeffs = dict(zip(teams, params[n_teams:2 * n_teams]))
        rho, gamma = params[-2:]

        log_likelihoods = [
            dc_log_like(
                row.home_team_goals,
                row.away_team_goals,
                attack_coeffs[row.home_team],
                defence_coeffs[row.home_team],
                attack_coeffs[row.away_team],
                defence_coeffs[row.away_team],
                rho,
                gamma
            )
            for row in dataset.itertuples()
        ]
        if weights is not None:
            log_likelihoods = np.array(log_likelihoods) * np.array(weights)
        return -np.sum(log_likelihoods)

    constraints = [
        {"type": "eq", "fun": lambda p, n=n_teams: np.sum(p[:n])},
        {"type": "eq", "fun": lambda p, n=n_teams: np.sum(p[n:2*n])},
    ]

    bounds = Bounds(
        [-np.inf] * (2 * n_teams) + [-1.0, -np.inf],
        [ np.inf] * (2 * n_teams) + [ 1.0,  np.inf]
    )

    opt_output = minimize(
        estimate_parameters,
        init_vals,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=options,
        **kwargs
    )

    return dict(
        zip(
            ["attack_" + team for team in teams] +
            ["defence_" + team for team in teams] +
            ["rho", "home_adv"],
            opt_output.x
        )
    )

def dixon_coles_simulate_match(params_dict, home_team, away_team, max_goals=10):
    def calc_means(param_dict, home_team, away_team):
        return [
            np.exp(param_dict["attack_" + home_team] + param_dict["defence_" + away_team] + param_dict["home_adv"]),
            np.exp(param_dict["defence_" + home_team] + param_dict["attack_" + away_team])
        ]
    team_avgs = calc_means(params_dict, home_team, away_team)
    team_pred = [[poisson.pmf(i, team_avg) for i in range(max_goals + 1)] for team_avg in team_avgs]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    correction_matrix = np.array([
        [rho_correction(h, a, team_avgs[0], team_avgs[1], params_dict["rho"]) for a in range(2)]
        for h in range(2)
    ])
    output_matrix[:2, :2] *= correction_matrix
    return output_matrix

@st.cache_data(show_spinner=False)
def solve_parameters_cached(dataset, weights=None):
    return solve_parameters(dataset, weights=weights)

@st.cache_data(show_spinner=False)
def dixon_coles_simulate_match_cached(params_dict, home_team, away_team, max_goals=10):
    return dixon_coles_simulate_match(params_dict, home_team, away_team, max_goals)

def run(country: str, league: str, season: str):
    match_df, historical_df = require_session_data("match_data", "tff_historical_matches")

    all_seasons = historical_df['season'].dropna().unique()
    selected_seasons = st.multiselect(
        "Select the season(s) to be used in the model:",
        options=all_seasons,
        default=list(all_seasons),
        key="dixon_coles_seasons"
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
        key="dixon_coles_selected_match"
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

        use_decay = st.checkbox(
            "Use time decay",
            value=False,
            help=(
                "If selected, recent matches will have more influence in the model, while older matches will have less. "
                "This helps the model reflect the current form of teams. If not selected, all matches are weighted equally, "
                "regardless of date."
            )
        )

        if use_decay:
            filtered_historical["date"] = pd.to_datetime(filtered_historical["date"], errors="coerce")
            xi = 0.0018
            max_date = filtered_historical["date"].max()
            days_diff = (max_date - filtered_historical["date"]).dt.days.values
            weights = np.exp(-xi * days_diff)
            weights = weights / np.mean(weights)
        else:
            weights = np.ones(len(filtered_historical))

        if st.button("Run Model", key="dixon_coles_calc_button"):
            with st.spinner("Running the model... The first run might take a bit longer while the system is being set up."):
                params = solve_parameters_cached(filtered_historical, weights=weights)
                model_df = dixon_coles_simulate_match_cached(params, home_team, away_team, max_goals=10)

            if model_df is not None and hasattr(model_df, "shape") and model_df.size > 0 and np.any(model_df):

                home_win_prob = np.sum(np.tril(model_df, -1)) * 100
                draw_prob = np.sum(np.diag(model_df)) * 100
                away_win_prob = np.sum(np.triu(model_df, 1)) * 100

                percentage_matrix = model_df * 100

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
                model_title = "Predicted by Dixon-Coles"
                if use_decay:
                    model_title += " (with Time Decay)"
                ax.set_title(
                    f"{season} {league} - Week {last_week} Scoreline Outcome Probabilities\n"
                    f"{model_title}",
                    fontsize=12,
                    fontweight="bold",
                    pad=30
                )
                ax.text(
                    0.5, 1.12,
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