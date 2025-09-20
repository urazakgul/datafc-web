import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

def _poisson_pmf(k, lam):
    k = np.asarray(k, dtype=int)
    lam = np.asarray(lam, dtype=float)
    return np.exp(-lam) * np.power(lam, k) / np.maximum(1, np.array([np.math.factorial(int(x)) for x in k]))

def _expected_points_from_lambdas(lam_for: float, lam_against: float, max_goals: int = 10) -> float:
    ks = np.arange(0, max_goals + 1)
    pf = _poisson_pmf(ks, lam_for)
    pa = _poisson_pmf(ks, lam_against)

    M = np.outer(pf, pa)
    p_draw = np.trace(M)
    p_loss = np.triu(M, k=1).sum()
    p_win  = np.tril(M, k=-1).sum()

    return 3.0 * p_win + 1.0 * p_draw

@st.cache_data(show_spinner=False)
def _preload_logos(team_logo_urls: dict, timeout: int = 10) -> dict:
    team_logo_images = {}
    for team, url in (team_logo_urls or {}).items():
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGBA")
            team_logo_images[team] = img
        except Exception:
            team_logo_images[team] = None
    return team_logo_images

def _add_team_marker(ax, x, y, img):
    if img is not None:
        ab = AnnotationBbox(
            OffsetImage(img, zoom=0.3),
            (x, y),
            frameon=False,
        )
        ax.add_artist(ab)
    else:
        ax.plot(x, y, "o", color="gray", markersize=6)

def run(country: str, league: str, season: str):
    match_df, shots_df, standings_df = require_session_data("match_data", "shots_data", "standings_data")

    match_df = filter_matches_by_status(match_df, "Ended")

    max_week = match_df["week"].max()

    standings_df = standings_df[standings_df["category"] == "Total"][["team_name", "points"]]

    match_cols = ["tournament", "season", "week", "game_id", "home_team", "away_team"]
    shots_df = shots_df.merge(
        match_df[match_cols],
        on=["tournament", "season", "week", "game_id"],
        how="inner"
    )

    is_home = shots_df.get("is_home")
    shots_df["team_name"] = np.where(is_home, shots_df["home_team"], shots_df["away_team"])
    shots_df["opponent_team_name"] = np.where(is_home, shots_df["away_team"], shots_df["home_team"])

    g_for = (
        shots_df
        .groupby(["tournament", "season", "week", "game_id", "team_name"], dropna=False)["xg"]
        .sum()
        .rename("lambda_for")
        .reset_index()
    )
    g_against = (
        shots_df
        .groupby(["tournament", "season", "week", "game_id", "opponent_team_name"], dropna=False)["xg"]
        .sum()
        .rename("lambda_against")
        .reset_index()
        .rename(columns={"opponent_team_name": "team_name"})
    )

    lambdas = g_for.merge(
        g_against,
        on=["tournament", "season", "week", "game_id", "team_name"],
        how="outer"
    ).fillna(0.0)

    lambdas["xpts_match"] = lambdas.apply(
        lambda r: _expected_points_from_lambdas(float(r["lambda_for"]), float(r["lambda_against"]), max_goals=10),
        axis=1
    )

    team_xpts_total = (
        lambdas.groupby(["team_name"], dropna=False)["xpts_match"]
        .sum()
        .rename("expected_points")
        .reset_index()
    )

    comparison_df = (
        team_xpts_total
        .merge(standings_df.rename(columns={"points": "actual_points"}), on="team_name", how="left")
        .sort_values("expected_points", ascending=False)
        .reset_index(drop=True)
    )

    comparison_df = comparison_df[["team_name", "expected_points", "actual_points"]]

    urls = st.session_state.get("team_logo_urls", {})
    if "team_logo_images" not in st.session_state:
        st.session_state["team_logo_images"] = _preload_logos(urls)
    team_logo_images = st.session_state["team_logo_images"]
    st.session_state.pop('team_logo_images', None)

    plot_df = comparison_df.dropna(subset=["expected_points", "actual_points"]).copy()
    x = plot_df["expected_points"].to_numpy(dtype=float)
    y = plot_df["actual_points"].to_numpy(dtype=float)

    max_xy = float(np.max([x.max() if x.size else 1.0, y.max() if y.size else 1.0]))
    lim = max(1.0, max_xy) * 1.08

    rng = np.random.default_rng(42)
    jitter_scale = 0.01 * lim
    jx = rng.normal(0.0, jitter_scale, size=x.shape)
    jy = rng.normal(0.0, jitter_scale, size=y.shape)

    fig, ax = plt.subplots(figsize=(11, 8))

    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=2, alpha=0.8, color="black")

    ax.fill_between(
        x=[0, lim],
        y1=[0, lim],
        y2=lim,
        color="green",
        alpha=0.1
    )

    ax.fill_between(
        x=[0, lim],
        y1=[0, lim],
        y2=0,
        color="red",
        alpha=0.1
    )

    for i, row in plot_df.iterrows():
        name = row["team_name"]
        img = team_logo_images.get(name)
        _add_team_marker(
            ax,
            float(row["expected_points"]) + jx[i],
            float(row["actual_points"]) + jy[i],
            img
        )

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Expected Points (xPts)", labelpad=20)
    ax.set_ylabel("Actual Points", labelpad=20)
    ax.set_title(
        f"{season} {league}\nExpected vs Actual Points\n(up to Week {max_week})",
        pad=35
    )

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="major", pad=10)

    over_patch  = mpatches.Patch(facecolor="green", alpha=0.2, label="Overperforming")
    under_patch = mpatches.Patch(facecolor="red",   alpha=0.2, label="Underperforming")

    plt.subplots_adjust(top=0.82)

    leg = ax.legend(
        handles=[over_patch, under_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        fontsize=10,
        frameon=False
    )

    st.pyplot(fig)