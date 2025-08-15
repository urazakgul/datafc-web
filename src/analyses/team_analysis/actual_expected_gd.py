import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import numpy as np
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

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

def _add_team_marker(ax, x, y, img, highlight: bool = False):
    if img is not None:
        zoom = 0.5 if highlight else 0.3
        ab = AnnotationBbox(
            OffsetImage(img, zoom=zoom),
            (x, y),
            frameon=highlight,
            bboxprops=(
                dict(edgecolor="red", linewidth=3, boxstyle="square,pad=0", facecolor="none")
                if highlight else None
            )
        )
        ax.add_artist(ab)
    else:
        ax.plot(
            x, y,
            "o",
            color=("red" if highlight else "gray"),
            markersize=(15 if highlight else 6),
        )

def run(team: str, country: str, league: str, season: str):
    match_df, shots_df, standings_df = require_session_data("match_data", "shots_data", "standings_data")

    match_df = filter_matches_by_status(match_df, "Ended")
    match_df = match_df[["tournament", "season", "week", "game_id", "home_team", "away_team"]]

    max_week = match_df["week"].max()

    shots_df = shots_df.merge(match_df, on=["tournament", "season", "week", "game_id"], how="inner")

    shots_df["team_name"] = np.where(shots_df["is_home"], shots_df["home_team"], shots_df["away_team"])
    xg_by_team = (
        shots_df.groupby(["game_id", "team_name"], as_index=False)["xg"]
        .sum()
        .rename(columns={"xg": "xg"})
    )

    teams_map_home = match_df[["game_id", "home_team", "away_team"]].rename(
        columns={"home_team": "team_name", "away_team": "opponent"}
    )
    teams_map_away = match_df[["game_id", "home_team", "away_team"]].rename(
        columns={"away_team": "team_name", "home_team": "opponent"}
    )
    teams_map = pd.concat([teams_map_home, teams_map_away], ignore_index=True)

    teams_xg = teams_map.merge(xg_by_team, on=["game_id", "team_name"], how="left")
    opp_xg = xg_by_team.rename(columns={"team_name": "opponent", "xg": "xga"})
    xg_xga_df = teams_xg.merge(opp_xg[["game_id", "opponent", "xga"]], on=["game_id", "opponent"], how="left")
    xg_xga_df[["xg", "xga"]] = xg_xga_df[["xg", "xga"]].fillna(0)

    team_totals_df = xg_xga_df.groupby("team_name", as_index=False)[["xg", "xga"]].sum()
    team_totals_df = team_totals_df.rename(columns={"xga": "xgConceded"})

    standings_df = standings_df[standings_df["category"] == "Total"][["team_name", "scores_for", "scores_against"]]

    actual_xg_xga_diffs = standings_df.merge(team_totals_df, on="team_name", how="inner")
    actual_xg_xga_diffs["xgDiff"] = actual_xg_xga_diffs["scores_for"] - actual_xg_xga_diffs["xg"]
    actual_xg_xga_diffs["xgConcededDiff"] = actual_xg_xga_diffs["scores_against"] - actual_xg_xga_diffs["xgConceded"]
    xg_xga_teams = actual_xg_xga_diffs[["team_name", "xgDiff", "xgConcededDiff"]].copy()

    if team not in set(xg_xga_teams["team_name"]):
        st.warning(f"No data available yet for {team} in {season} {league}.")
        return

    urls = st.session_state.get("team_logo_urls", {})
    if "team_logo_images" not in st.session_state:
        st.session_state["team_logo_images"] = _preload_logos(urls)
    team_logo_images = st.session_state["team_logo_images"]

    x = xg_xga_teams["xgDiff"].to_numpy(dtype=float)
    y = xg_xga_teams["xgConcededDiff"].to_numpy(dtype=float)
    max_abs_x = float(np.max(np.abs(x))) if x.size > 0 else 1.0
    max_abs_y = float(np.max(np.abs(y))) if y.size > 0 else 1.0
    lim_x = max_abs_x * 1.05
    lim_y = max_abs_y * 1.05

    jitter_pct = 1.0
    rng = np.random.default_rng(42)
    jitter_x = rng.normal(0.0, (jitter_pct / 100.0) * (2 * lim_x), size=x.shape)
    jitter_y = rng.normal(0.0, (jitter_pct / 100.0) * (2 * lim_y), size=y.shape)

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.axhline(y=0, color="darkred", linestyle="--", linewidth=2)
    ax.axvline(x=0, color="darkblue", linestyle="--", linewidth=2)

    for i, row in xg_xga_teams.iterrows():
        name = row["team_name"]
        img = team_logo_images.get(name)
        xi, yi = float(row["xgDiff"]), float(row["xgConcededDiff"])
        if name == team:
            _add_team_marker(ax, xi, yi, img, highlight=True)
        else:
            _add_team_marker(ax, xi + jitter_x[i], yi + jitter_y[i], img, highlight=False)

    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_y, lim_y)
    ax.invert_yaxis()

    ax.tick_params(axis="both", which="major", pad=14)

    ax.set_title(
        f"{season} {league}\nActual vs Expected Goal Differences by Team, with {team} highlighted\n"
        f"(up to Week {max_week})",
        fontsize=16,
        fontweight="bold",
        pad=30,
        loc="center"
    )
    ax.set_xlabel("Actual - xG (higher is better)", labelpad=20)
    ax.set_ylabel("Actual - xGA (lower is better)", labelpad=20)
    ax.grid(True, linestyle="--", alpha=0.7)

    st.pyplot(fig)