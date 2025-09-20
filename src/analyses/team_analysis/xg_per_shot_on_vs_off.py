import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import pandas as pd
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import FormatStrFormatter
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
            OffsetImage(img, zoom=0.3),
            (x, y),
            frameon=False,
        )
        ax.add_artist(ab)
    else:
        ax.plot(x, y, "o", color="gray", markersize=6)

def run(country: str, league: str, season: str):
    match_df, shots_df = require_session_data("match_data", "shots_data")
    match_df = filter_matches_by_status(match_df, "Ended")

    match_cols = ["tournament", "season", "week", "game_id", "home_team", "away_team"]
    shots_df = shots_df.merge(match_df[match_cols], on=["tournament", "season", "week", "game_id"], how="left")

    max_week = match_df["week"].max()

    is_home = shots_df.get("is_home")
    if is_home is None:
        is_home = np.zeros(len(shots_df), dtype=bool)
    else:
        is_home = is_home.fillna(False).astype(bool)

    shots_df["team_name"] = np.where(is_home, shots_df["home_team"], shots_df["away_team"])

    shots_df["shot_type"] = shots_df["shot_type"].astype(str).str.strip().str.lower()
    shots_df["situation"] = shots_df["situation"].astype(str).str.strip().str.lower()

    shots_df = shots_df[shots_df["situation"] != "penalty"].copy()

    on_target_set = {"goal", "save"}
    off_target_set = {"miss", "post", "block"}

    shots_df["is_on_target"] = shots_df["shot_type"].isin(on_target_set)
    shots_df["is_off_target"] = shots_df["shot_type"].isin(off_target_set)

    filtered_shots_df = shots_df[shots_df["is_on_target"] | shots_df["is_off_target"]].copy()
    filtered_shots_df["xg"] = pd.to_numeric(filtered_shots_df["xg"], errors="coerce").fillna(0.0)

    agg = filtered_shots_df.groupby("team_name").apply(
        lambda df: pd.Series({
            "shots_on": int(df["is_on_target"].sum()),
            "shots_off": int(df["is_off_target"].sum()),
            "xg_on_sum": df.loc[df["is_on_target"], "xg"].sum(),
            "xg_off_sum": df.loc[df["is_off_target"], "xg"].sum(),
        })
    ).reset_index()

    agg["xg_per_shot_on"] = np.where(agg["shots_on"] > 0, agg["xg_on_sum"] / agg["shots_on"], np.nan)
    agg["xg_per_shot_off"] = np.where(agg["shots_off"] > 0, agg["xg_off_sum"] / agg["shots_off"], np.nan)

    cols = [
        "team_name",
        "shots_on", "xg_on_sum", "xg_per_shot_on",
        "shots_off", "xg_off_sum", "xg_per_shot_off",
    ]
    agg = agg[cols].sort_values("xg_per_shot_on", ascending=False).round(3)

    teams_df = agg[["team_name", "xg_per_shot_on", "xg_per_shot_off"]].copy()
    teams_df = teams_df.rename(columns={
        "xg_per_shot_on": "On-Target xG/Shot",
        "xg_per_shot_off": "Off-Target xG/Shot",
    })
    teams_df = teams_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["On-Target xG/Shot", "Off-Target xG/Shot"]
    )

    urls = st.session_state.get("team_logo_urls", {})
    if "team_logo_images" not in st.session_state:
        st.session_state["team_logo_images"] = _preload_logos(urls)
    team_logo_images = st.session_state["team_logo_images"]
    st.session_state.pop("team_logo_images", None)

    x_col = "Off-Target xG/Shot"
    y_col = "On-Target xG/Shot"
    x = teams_df[x_col].to_numpy(dtype=float)
    y = teams_df[y_col].to_numpy(dtype=float)

    max_x, min_x = (float(np.nanmax(x)), float(np.nanmin(x))) if x.size else (1.0, 0.0)
    max_y, min_y = (float(np.nanmax(y)), float(np.nanmin(y))) if y.size else (1.0, 0.0)
    pad_x = (max_x - min_x) * 0.05 if (max_x > min_x) else 0.05
    pad_y = (max_y - min_y) * 0.05 if (max_y > min_y) else 0.05

    x_med = float(np.nanmedian(x)) if x.size else 0.0
    y_med = float(np.nanmedian(y)) if y.size else 0.0

    jitter_pct = 1.0
    rng = np.random.default_rng(42)
    jitter_x = rng.normal(0.0, (jitter_pct / 100.0) * max(1e-9, (max_x - min_x)), size=x.shape)
    jitter_y = rng.normal(0.0, (jitter_pct / 100.0) * max(1e-9, (max_y - min_y)), size=y.shape)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axvline(x=x_med, color="darkblue", linestyle="--", linewidth=2)
    ax.axhline(y=y_med, color="darkred", linestyle="--", linewidth=2)

    for i, row in teams_df.reset_index(drop=True).iterrows():
        name = row["team_name"]
        img = team_logo_images.get(name)
        xi, yi = float(row[x_col]), float(row[y_col])
        _add_team_marker(ax, xi + jitter_x[i], yi + jitter_y[i], img)

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

    ax.tick_params(axis="both", which="major", pad=14)
    ax.set_title(
        f"{season} {league}\nNon-Penalty On-Target vs Off-Target xG per Shot\n(up to Week {max_week})",
        fontsize=16, fontweight="bold", pad=30, loc="center"
    )
    ax.set_xlabel(x_col, labelpad=20)
    ax.set_ylabel(y_col, labelpad=20)
    ax.grid(True, linestyle="--", alpha=0.7)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    st.pyplot(fig)