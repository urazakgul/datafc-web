import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO
import requests

plt.style.use("fivethirtyeight")

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

@st.cache_data(show_spinner=False)
def preload_logos(team_logo_urls):
    team_logo_images = {}
    for team, url in team_logo_urls.items():
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGBA")
            team_logo_images[team] = img
        except Exception as e:
            print(f"Logo load error for {team}: {e}")
            team_logo_images[team] = None
    return team_logo_images

def getImageFromImage(img, zoom=0.3):
    if img is None:
        return None
    return OffsetImage(img, zoom=zoom, alpha=1)

def run(country: str, league: str, season: str):
    match_df, momentum_df = require_session_data("match_data", "momentum_data")

    match_df = filter_matches_by_status(match_df, "Ended")
    match_df = match_df[["country","tournament","season","week","game_id","home_team","away_team"]]

    max_week = match_df["week"].max()

    match_momentum_df = momentum_df.merge(
        match_df,
        on=["country","tournament","season","week","game_id"],
        how="left"
    )

    match_momentum_df.sort_values(by=["game_id", "minute"], inplace=True)

    match_momentum_df["home_team_momentum_count"] = match_momentum_df.groupby(
        ["country", "tournament", "season", "week", "game_id", "home_team"]
    )["value"].transform(lambda x: (x > 0).astype(int).cumsum())

    match_momentum_df["away_team_momentum_count"] = match_momentum_df.groupby(
        ["country", "tournament", "season", "week", "game_id", "away_team"]
    )["value"].transform(lambda x: (x < 0).astype(int).cumsum())

    match_momentum_df["home_team_momentum_value"] = match_momentum_df.groupby(
        ["country", "tournament", "season", "week", "game_id", "home_team"]
    )["value"].transform(lambda x: x.where(x > 0, 0).abs().cumsum())

    match_momentum_df["away_team_momentum_value"] = match_momentum_df.groupby(
        ["country", "tournament", "season", "week", "game_id", "away_team"]
    )["value"].transform(lambda x: x.where(x < 0, 0).abs().cumsum())

    home_last_momentum = match_momentum_df.groupby(
        ["country", "tournament", "season", "week", "home_team"]
    ).last().reset_index()[[
        "country", "tournament", "season", "week",
        "home_team", "home_team_momentum_count", "home_team_momentum_value",
        "away_team", "away_team_momentum_count", "away_team_momentum_value"
    ]]

    away_last_momentum = match_momentum_df.groupby(
        ["country", "tournament", "season", "week", "away_team"]
    ).last().reset_index()[[
        "country", "tournament", "season", "week",
        "away_team", "away_team_momentum_count", "away_team_momentum_value",
        "home_team", "home_team_momentum_count", "home_team_momentum_value"
    ]]

    home_last_momentum.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_team_momentum_count": "team_momentum_count",
        "home_team_momentum_value": "team_momentum_value",
        "away_team_momentum_count": "opponent_momentum_count",
        "away_team_momentum_value": "opponent_momentum_value"
    }, inplace=True)

    away_last_momentum.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_team_momentum_count": "team_momentum_count",
        "away_team_momentum_value": "team_momentum_value",
        "home_team_momentum_count": "opponent_momentum_count",
        "home_team_momentum_value": "opponent_momentum_value"
    }, inplace=True)

    final_momentum_df = pd.concat([home_last_momentum, away_last_momentum], ignore_index=True)

    final_summary_df = final_momentum_df.groupby(
        ["country", "tournament", "season", "team"]
    ).agg(
        total_team_momentum_count=("team_momentum_count", "sum"),
        total_team_momentum_value=("team_momentum_value", "sum"),
        total_opponent_momentum_count=("opponent_momentum_count", "sum"),
        total_opponent_momentum_value=("opponent_momentum_value", "sum")
    ).reset_index()

    final_summary_df["team_momentum_per"] = final_summary_df["total_team_momentum_value"] / final_summary_df["total_team_momentum_count"]
    final_summary_df["opponent_momentum_per"] = final_summary_df["total_opponent_momentum_value"] / final_summary_df["total_opponent_momentum_count"]

    if "team_logo_images" not in st.session_state:
        st.session_state["team_logo_images"] = preload_logos(st.session_state["team_logo_urls"])
    team_logo_images = st.session_state["team_logo_images"]

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(
        final_summary_df["team_momentum_per"],
        final_summary_df["opponent_momentum_per"],
        alpha=0
    )

    mean_team_momentum_per = final_summary_df["team_momentum_per"].mean()
    mean_opponent_momentum_per = final_summary_df["opponent_momentum_per"].mean()

    ax.axvline(x=mean_team_momentum_per, color="darkblue", linestyle="--", linewidth=2, label="League Avg (Teams)")
    ax.axhline(y=mean_opponent_momentum_per, color="darkred", linestyle="--", linewidth=2, label="League Avg (Opponents)")

    for _, row in final_summary_df.iterrows():
        team_name = row["team"]
        logo_img = team_logo_images.get(team_name, None)
        offset_img = getImageFromImage(logo_img, zoom=0.3)
        if offset_img is not None:
            ab = AnnotationBbox(offset_img, (row["team_momentum_per"], row["opponent_momentum_per"]), frameon=False)
            ax.add_artist(ab)
        else:
            ax.plot(row["team_momentum_per"], row["opponent_momentum_per"], "o", color="gray")

    ax.set_xlabel("Team Momentum Productivity (higher is better)", labelpad=20, fontsize=12)
    ax.set_ylabel("Opponent Momentum Productivity (lower is better)", labelpad=20, fontsize=12)
    ax.set_title(
        f"{season} {league}\nTeam vs Opponent Momentum Productivity\n(up to Week {max_week})",
        fontsize=16,
        fontweight="bold",
        pad=40
    )
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.invert_yaxis()

    st.pyplot(fig)