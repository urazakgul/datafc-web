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

def _clean_percent_columns(dataframe, columns_to_check, target_columns):
    for index, row in dataframe.iterrows():
        if any(keyword in row["stat_name"] for keyword in columns_to_check):
            for col in target_columns:
                dataframe.at[index, col] = row[col].replace("%", "").strip()
    return dataframe

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
    match_df, match_stats_df = require_session_data("match_data", "match_stats_data")

    match_df = filter_matches_by_status(match_df, "Ended")

    max_week = match_df["week"].max()

    match_df = match_df[["game_id", "home_team", "away_team"]]
    match_stats_df = match_stats_df[match_stats_df["period"] == "ALL"]

    percent_keywords = ["Ball possession"]
    target_columns = ["home_team_stat", "away_team_stat"]

    match_stats_df = _clean_percent_columns(match_stats_df, percent_keywords, target_columns)

    master_df = match_stats_df.merge(match_df, on="game_id")

    master_df = master_df[
        master_df["stat_name"].isin(["Ball possession", "Touches in penalty area"])
    ]

    all_stats_df_list = []
    for stat in master_df["stat_name"].unique():
        stat_df = master_df[master_df["stat_name"] == stat]
        temp_df = pd.DataFrame({
            "team_name": pd.concat([stat_df["home_team"], stat_df["away_team"]]),
            "stat_name": [stat] * len(stat_df) * 2,
            "stat_value": pd.concat([stat_df["home_team_stat"], stat_df["away_team_stat"]])
        })
        all_stats_df_list.append(temp_df)
    result_all_stats_df = pd.concat(all_stats_df_list, ignore_index=True).reset_index(drop=True)
    result_all_stats_df["stat_value"] = pd.to_numeric(result_all_stats_df["stat_value"], errors="coerce")

    poss_df = (
        result_all_stats_df[result_all_stats_df["stat_name"] == "Ball possession"]
        .groupby("team_name", as_index=False)["stat_value"].mean()
        .rename(columns={"stat_value": "Ball Possession (%) (avg)"})
    )
    poss_df["Ball Possession (%) (avg)"] = poss_df["Ball Possession (%) (avg)"].round(2)

    touches_df = (
        result_all_stats_df[result_all_stats_df["stat_name"] == "Touches in penalty area"]
        .groupby("team_name", as_index=False)["stat_value"].mean()
        .rename(columns={"stat_value": "Touches in Box (avg)"})
    )
    touches_df["Touches in Box (avg)"] = touches_df["Touches in Box (avg)"].round(2)

    teams_df = poss_df.merge(touches_df, on="team_name", how="inner")

    urls = st.session_state.get("team_logo_urls", {})
    if "team_logo_images" not in st.session_state:
        st.session_state["team_logo_images"] = _preload_logos(urls)
    team_logo_images = st.session_state["team_logo_images"]
    st.session_state.pop('team_logo_images', None)

    x_col = "Ball Possession (%) (avg)"
    y_col = "Touches in Box (avg)"
    x = teams_df[x_col].to_numpy(dtype=float)
    y = teams_df[y_col].to_numpy(dtype=float)

    max_x, min_x = (float(np.max(x)), float(np.min(x))) if x.size else (100.0, 0.0)
    max_y, min_y = (float(np.max(y)), float(np.min(y))) if y.size else (1.0, 0.0)
    pad_x = (max_x - min_x) * 0.05 if (max_x > min_x) else 1.0
    pad_y = (max_y - min_y) * 0.05 if (max_y > min_y) else 1.0

    jitter_pct = 1.0
    rng = np.random.default_rng(42)
    jitter_x = rng.normal(0.0, (jitter_pct / 100.0) * (max_x - min_x + 1e-9), size=x.shape)
    jitter_y = rng.normal(0.0, (jitter_pct / 100.0) * (max_y - min_y + 1e-9), size=y.shape)

    x_med = float(np.nanmedian(x)) if x.size else 50.0
    y_med = float(np.nanmedian(y)) if y.size else 0.0

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axvline(x=x_med, color="darkblue", linestyle="--", linewidth=2)
    ax.axhline(y=y_med, color="darkred", linestyle="--", linewidth=2)

    for i, row in teams_df.iterrows():
        name = row["team_name"]
        img = team_logo_images.get(name)
        xi, yi = float(row[x_col]), float(row[y_col])
        _add_team_marker(ax, xi + jitter_x[i], yi + jitter_y[i], img)

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

    ax.tick_params(axis="both", which="major", pad=14)
    ax.set_title(
        f"{season} {league}\nPossession vs Touches in Box\n(up to Week {max_week})",
        fontsize=16,
        fontweight="bold",
        pad=30,
        loc="center"
    )
    ax.set_xlabel(x_col, labelpad=20)
    ax.set_ylabel(y_col, labelpad=20)
    ax.grid(True, linestyle="--", alpha=0.7)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    st.pyplot(fig)