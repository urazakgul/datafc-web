import numpy as np
import pandas as pd
from scipy.stats import skellam
from scipy.optimize import minimize, Bounds

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