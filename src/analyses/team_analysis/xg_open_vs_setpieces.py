import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import pandas as pd
import numpy as np
from matplotlib.ticker import FormatStrFormatter
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

def _add_team_marker(ax, x, y, img):
    if img is not None:
        ab = AnnotationBbox(
            OffsetImage(img, zoom=0.28),
            (x, y),
            frameon=False,
        )
        ax.add_artist(ab)
    else:
        ax.plot(x, y, "o", color="gray", markersize=6)

OPEN_PLAY = {"regular", "fast-break"}
SET_PIECES_ALL = {"set-piece", "corner", "free-kick", "throw-in-set-piece", "penalty"}

def _play_type(situation: str, exclude_penalties: bool) -> str | None:
    if situation is None:
        return None
    s = str(situation).strip().lower()
    if s in OPEN_PLAY:
        return "Open Play"
    if s in SET_PIECES_ALL:
        if exclude_penalties and s == "penalty":
            return None
        return "Set Pieces"
    return None

def run(country: str, league: str, season: str):
    match_df, shots_df = require_session_data("match_data", "shots_data")
    match_df = filter_matches_by_status(match_df, "Ended")

    max_week = match_df["week"].max()

    match_df = match_df[["game_id", "season", "week", "home_team", "away_team"]]
    shots_df = shots_df.merge(
        match_df[["game_id", "home_team", "away_team"]],
        on="game_id",
        how="left"
    )

    shots_df["team_name"] = np.where(shots_df["is_home"], shots_df["home_team"], shots_df["away_team"])

    shots_df = shots_df[shots_df["goal_type"] != "own"]

    exclude_penalties = st.checkbox("Exclude penalties from Set Pieces", value=False)

    shots_df["play_type"] = shots_df["situation"].apply(
        lambda s: _play_type(s, exclude_penalties)
    )
    shots_df = shots_df.dropna(subset=["play_type"])

    team_games = pd.concat([
        match_df[["home_team"]].rename(columns={"home_team": "team_name"}),
        match_df[["away_team"]].rename(columns={"away_team": "team_name"})
    ], ignore_index=True)
    games_per_team = (
        team_games.value_counts("team_name")
        .rename_axis("team_name")
        .reset_index(name="matches_played")
    )

    xg_by_team_type = (
        shots_df.groupby(["team_name", "play_type"], as_index=False)
                .agg(xg_sum=("xg", "sum"))
    )

    xg_wide = xg_by_team_type.pivot(index="team_name", columns="play_type", values="xg_sum").fillna(0.0)
    for col in ["Open Play", "Set Pieces"]:
        if col not in xg_wide.columns:
            xg_wide[col] = 0.0
    xg_wide = xg_wide.reset_index()

    xg_pm = xg_wide.merge(games_per_team, on="team_name", how="left")
    xg_pm["matches_played"] = xg_pm["matches_played"].replace(0, np.nan)
    xg_pm["Open Play xG"] = (xg_pm["Open Play"] / xg_pm["matches_played"]).round(3)
    xg_pm["Set Pieces xG"] = (xg_pm["Set Pieces"] / xg_pm["matches_played"]).round(3)

    urls = st.session_state.get("team_logo_urls", {})
    logos = _preload_logos(urls)

    x_col = "Open Play xG"
    y_col = "Set Pieces xG"
    plot_df = xg_pm[["team_name", x_col, y_col]].dropna()

    x = plot_df[x_col].to_numpy(dtype=float)
    y = plot_df[y_col].to_numpy(dtype=float)

    rng = np.random.default_rng(123)
    jitter_x = rng.normal(0.0, 0.01, size=x.shape)
    jitter_y = rng.normal(0.0, 0.01, size=y.shape)

    x_med = float(np.nanmedian(x)) if x.size else 0.0
    y_med = float(np.nanmedian(y)) if y.size else 0.0

    max_x, min_x = (float(np.max(x)), float(np.min(x))) if x.size else (1.0, 0.0)
    max_y, min_y = (float(np.max(y)), float(np.min(y))) if y.size else (1.0, 0.0)
    pad_x = max(0.05, (max_x - min_x) * 0.10)
    pad_y = max(0.05, (max_y - min_y) * 0.10)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axvline(x=x_med, color="darkblue", linestyle="--", linewidth=2)
    ax.axhline(y=y_med, color="darkred", linestyle="--", linewidth=2)

    for i, row in plot_df.iterrows():
        name = row["team_name"]
        img = logos.get(name)
        xi, yi = float(row[x_col]), float(row[y_col])
        _add_team_marker(ax, xi + jitter_x[i], yi + jitter_y[i], img)

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

    ax.tick_params(axis="both", which="major", pad=14)
    excl_txt = " (penalties excluded)" if exclude_penalties else ""
    ax.set_title(
        f"{season} {league}\nOpen Play vs Set Pieces xG per Match{excl_txt}\n(up to Week {max_week})",
        fontsize=16,
        fontweight="bold",
        pad=30,
        loc="center"
    )
    ax.set_xlabel(x_col, labelpad=20)
    ax.set_ylabel(y_col, labelpad=20)
    ax.grid(True, linestyle="--", alpha=0.7)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    st.pyplot(fig)