import streamlit as st
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize, Bounds

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

def solve_parameters(dataset, init_vals=None, options={"disp": False, "maxiter": 100}, **kwargs):
    teams = np.sort(
        list(
            set(dataset["home_team"].unique()) | set(dataset["away_team"].unique())
        )
    )
    n_teams = len(teams)

    if init_vals is None:
        avg_attack = dataset.groupby("home_team")["home_team_goals"].mean().reindex(teams).fillna(1.0).values
        avg_defence = -dataset.groupby("away_team")["away_team_goals"].mean().reindex(teams).fillna(1.0).values
        init_vals = np.concatenate([
            avg_attack,
            avg_defence,
            np.array([0, 1.0])
        ])

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
                rho, gamma
            )
            for row in dataset.itertuples()
        ]
        return -np.sum(log_likelihoods)

    constraints = [{"type": "eq", "fun": lambda x, n=n_teams: sum(x[:n]) - n}]

    bounds = Bounds(
        [-np.inf] * n_teams + [-np.inf] * n_teams + [-1, 0],
        [np.inf] * n_teams + [np.inf] * n_teams + [1, np.inf]
    )

    opt_output = minimize(estimate_parameters, init_vals, options=options, constraints=constraints, bounds=bounds, **kwargs)

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
def solve_parameters_cached(dataset):
    return solve_parameters(dataset)

@st.cache_data(show_spinner=False)
def dixon_coles_simulate_match_cached(params_dict, home_team, away_team, max_goals=10):
    return dixon_coles_simulate_match(params_dict, home_team, away_team, max_goals)