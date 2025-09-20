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
            OffsetImage(img, zoom=0.3),
            (x, y),
            frameon=False,
        )
        ax.add_artist(ab)
    else:
        ax.plot(x, y, "o", color="gray", markersize=6)

def _safe_avg(total, n):
    return (total / n) if (n and n > 0) else np.nan

def _compute_game_stats(g: pd.DataFrame) -> pd.Series:
    home_first_sub = g.loc[g["home_sub_flag"].eq(1), "minute_slot"].min()
    away_first_sub = g.loc[g["away_sub_flag"].eq(1), "minute_slot"].min()

    if pd.notna(home_first_sub):
        mh_before = g["minute_f"] < home_first_sub
        mh_after  = g["minute_f"] >= home_first_sub
    else:
        mh_before = pd.Series(True, index=g.index)
        mh_after  = pd.Series(False, index=g.index)

    home_minutes_before = int(mh_before.sum())
    home_minutes_after = int(mh_after.sum())

    sum_h_before = g.loc[mh_before & (g["value"] > 0), "value"].sum()
    sum_h_after = g.loc[mh_after  & (g["value"] > 0), "value"].sum()

    home_avg_before = _safe_avg(sum_h_before, home_minutes_before)
    home_avg_after = _safe_avg(sum_h_after,  home_minutes_after)
    home_diff = (home_avg_after - home_avg_before) if pd.notna(home_avg_before) else np.nan

    if pd.notna(away_first_sub):
        ma_before = g["minute_f"] < away_first_sub
        ma_after = g["minute_f"] >= away_first_sub
    else:
        ma_before = pd.Series(True, index=g.index)
        ma_after = pd.Series(False, index=g.index)

    away_minutes_before = int(ma_before.sum())
    away_minutes_after = int(ma_after.sum())

    sum_a_before = (-g.loc[ma_before & (g["value"] < 0), "value"]).sum()
    sum_a_after = (-g.loc[ma_after  & (g["value"] < 0), "value"]).sum()

    away_avg_before = _safe_avg(sum_a_before, away_minutes_before)
    away_avg_after = _safe_avg(sum_a_after,  away_minutes_after)
    away_diff = (away_avg_after - away_avg_before) if pd.notna(away_avg_before) else np.nan

    return pd.Series({
        "home_team": g["home_team"].iloc[0],
        "away_team": g["away_team"].iloc[0],
        "home_first_sub_min": home_first_sub,
        "away_first_sub_min": away_first_sub,
        "home_diff": home_diff,
        "away_diff": away_diff,
        "home_minutes_before": home_minutes_before,
        "home_minutes_after": home_minutes_after,
        "away_minutes_before": away_minutes_before,
        "away_minutes_after": away_minutes_after,
        "home_sum_before": sum_h_before,
        "home_sum_after": sum_h_after,
        "away_sum_before": sum_a_before,
        "away_sum_after": sum_a_after,
    })

def run(country: str, league: str, season: str):
    match_df, momentum_df, substitutions_df, lineups_df = require_session_data(
        "match_data", "momentum_data", "substitutions_data", "lineups_data"
    )

    match_df = filter_matches_by_status(match_df, "Ended")
    match_df = match_df[["country","tournament","season","week","game_id","home_team","away_team"]]

    max_week = match_df["week"].max()

    momentum_df = momentum_df.copy()
    momentum_df["minute_f"] = pd.to_numeric(momentum_df["minute"], errors="coerce")
    momentum_df["minute_slot"] = np.floor(momentum_df["minute_f"]).astype("Int64")

    match_momentum_df = (
        momentum_df.merge(
            match_df,
            on=["country","tournament","season","week","game_id"],
            how="left"
        )
        .sort_values(["game_id", "minute_f"])
        .reset_index(drop=True)
    )

    subs = substitutions_df.copy()
    subs["minute_int"] = pd.to_numeric(subs.get("time"), errors="coerce").astype("Int64")

    subs = subs.merge(
        match_df[["game_id","home_team","away_team"]],
        on="game_id",
        how="left"
    )

    ply_to_side = (
        lineups_df[["game_id","team","player_id"]]
        .dropna(subset=["player_id"])
        .drop_duplicates()
    )

    subs = subs.merge(
        ply_to_side.rename(columns={"team":"in_side", "player_id":"player_in_id"}),
        on=["game_id","player_in_id"], how="left"
    ).merge(
        ply_to_side.rename(columns={"team":"out_side", "player_id":"player_out_id"}),
        on=["game_id","player_out_id"], how="left"
    )

    subs["side"] = subs["out_side"].fillna(subs["in_side"])
    subs["is_home_sub"] = (subs["side"] == "home").astype(int)
    subs["is_away_sub"] = (subs["side"] == "away").astype(int)

    home_flags = (
        subs.loc[subs["is_home_sub"].eq(1), ["game_id","minute_int"]]
            .dropna(subset=["minute_int"]).drop_duplicates()
            .rename(columns={"minute_int":"minute_slot"})
            .assign(home_sub_flag=1)
    )
    away_flags = (
        subs.loc[subs["is_away_sub"].eq(1), ["game_id","minute_int"]]
            .dropna(subset=["minute_int"]).drop_duplicates()
            .rename(columns={"minute_int":"minute_slot"})
            .assign(away_sub_flag=1)
    )

    match_momentum_df = (
        match_momentum_df
          .merge(home_flags, on=["game_id","minute_slot"], how="left")
          .merge(away_flags, on=["game_id","minute_slot"], how="left")
    )

    match_momentum_df["home_sub_flag"] = match_momentum_df["home_sub_flag"].fillna(0).astype(int)
    match_momentum_df["away_sub_flag"] = match_momentum_df["away_sub_flag"].fillna(0).astype(int)

    game_momentum_change = (
        match_momentum_df
          .groupby("game_id", as_index=True)
          .apply(_compute_game_stats)
          .reset_index()
          .sort_values("game_id")
    )

    home_part = game_momentum_change[["game_id","home_team","home_first_sub_min","home_diff"]].rename(
        columns={"home_team":"team", "home_first_sub_min":"first_sub_min", "home_diff":"momentum_diff"}
    )
    home_part["side"] = "home"

    away_part = game_momentum_change[["game_id","away_team","away_first_sub_min","away_diff"]].rename(
        columns={"away_team":"team", "away_first_sub_min":"first_sub_min", "away_diff":"momentum_diff"}
    )
    away_part["side"] = "away"

    team_long = pd.concat([home_part, away_part], ignore_index=True)

    team_stats = (
        team_long
        .groupby("team", as_index=False)
        .agg(
            avg_first_sub_min=("first_sub_min", lambda s: s.dropna().mean() if len(s.dropna()) else np.nan),
            avg_momentum_change=("momentum_diff", lambda s: s.dropna().mean() if len(s.dropna()) else np.nan),
            n_games=("game_id", "nunique"),
            n_games_with_sub=("first_sub_min", lambda s: s.notna().sum()),
        )
        .sort_values("team")
    )

    urls = st.session_state.get("team_logo_urls", {})
    if "team_logo_images" not in st.session_state:
        st.session_state["team_logo_images"] = _preload_logos(urls)
    team_logo_images = st.session_state["team_logo_images"]

    teams_df = (
        team_stats
        .rename(columns={"team": "team_name"})
        .dropna(subset=["avg_first_sub_min", "avg_momentum_change"])
        .reset_index(drop=True)
    )

    x_col = "avg_first_sub_min"
    y_col = "avg_momentum_change"

    x = teams_df[x_col].to_numpy(dtype=float)
    y = teams_df[y_col].to_numpy(dtype=float)

    max_x, min_x = (float(np.nanmax(x)), float(np.nanmin(x))) if x.size else (100.0, 0.0)
    max_y, min_y = (float(np.nanmax(y)), float(np.nanmin(y))) if y.size else (1.0, 0.0)
    pad_x = (max_x - min_x) * 0.05 if (max_x > min_x) else 1.0
    pad_y = (max_y - min_y) * 0.10 if (max_y > min_y) else 0.5

    jitter_pct = 1.0
    rng = np.random.default_rng(42)
    jitter_x = rng.normal(0.0, (jitter_pct / 100.0) * (max_x - min_x + 1e-9), size=x.shape)
    jitter_y = rng.normal(0.0, (jitter_pct / 100.0) * (max_y - min_y + 1e-9), size=y.shape)

    x_med = float(np.nanmedian(x)) if x.size else 0.0
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
        f"{season} {league}\nFirst Sub Minute vs Momentum Change\n(up to Week {max_week})",
        fontsize=16, fontweight="bold", pad=30, loc="center"
    )
    ax.set_xlabel("Average First Substitution Minute", labelpad=20)
    ax.set_ylabel("Average Momentum Change (after − before)", labelpad=20)
    ax.grid(True, linestyle="--", alpha=0.7)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    st.pyplot(fig)