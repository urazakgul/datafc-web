import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize, Bounds

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