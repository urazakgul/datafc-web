import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances
from matplotlib import cm, colors as mcolors
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

CATEGORIES = {
    "Attack": [
        "Big chances scored", "Big chances missed", "Through balls",
        "Touches in penalty area", "Fouled in final third", "Offsides",
    ],
    "Defending": [
        "Tackles won", "Total tackles", "Interceptions", "Recoveries",
        "Clearances", "Errors lead to a shot", "Errors lead to a goal",
    ],
    "Duels": [
        "Duels", "Dispossessed", "Ground duels", "Aerial duels", "Dribbles",
    ],
    "Goalkeeping": [
        "Total saves", "Goals prevented", "High claims", "Punches",
        "Goal kicks", "Big saves", "Penalty saves",
    ],
    "Passes": [
        "Accurate passes", "Throw-ins", "Final third entries", "Final third phase",
        "Long balls", "Crosses",
    ],
    "Shots": [
        "Total shots", "Shots on target", "Hit woodwork", "Shots off target",
        "Blocked shots", "Shots inside box", "Shots outside box",
    ],
    "Match overview": [
        "Ball possession", "Expected goals", "Big chances", "Total shots",
        "Goalkeeper saves", "Corner kicks", "Fouls", "Passes", "Tackles",
        "Free kicks", "Yellow cards", "Red cards",
    ],
}

def _clean_percent_columns(dataframe, columns_to_check, target_columns):
    for index, row in dataframe.iterrows():
        if any(keyword in row["stat_name"] for keyword in columns_to_check):
            for col in target_columns:
                dataframe.at[index, col] = row[col].replace("%", "").strip()
    return dataframe

def _clean_parenthesis_columns(dataframe, columns_to_check, target_columns):
    for index, row in dataframe.iterrows():
        if any(keyword in row["stat_name"] for keyword in columns_to_check):
            for col in target_columns:
                if "(" in row[col]:
                    dataframe.at[index, col] = row[col].split("(")[0].strip()
    return dataframe

def run(team: str, country: str, league: str, season: str):
    match_df, match_stats_df = require_session_data("match_data", "match_stats_data")

    match_df = filter_matches_by_status(match_df, "Ended")

    max_week = match_df["week"].max()

    match_df = match_df[["game_id", "home_team", "away_team"]]

    match_stats_df = match_stats_df[match_stats_df["period"] == "ALL"]

    percent_keywords = ["Ball possession", "Tackles won", "Duels"]
    parenthesis_keywords = ["Final third phase", "Long balls", "Crosses", "Ground duels", "Aerial duels", "Dribbles"]
    target_columns = ["home_team_stat", "away_team_stat"]

    match_stats_df = _clean_percent_columns(match_stats_df, percent_keywords, target_columns)
    match_stats_df = _clean_parenthesis_columns(match_stats_df, parenthesis_keywords, target_columns)

    master_df = match_stats_df.merge(match_df, on="game_id")

    non_overview_stats = set().union(*[set(v) for k, v in CATEGORIES.items() if k != "Match overview"])
    overview_stats = set(CATEGORIES["Match overview"])
    keep_overview_only = sorted(list(overview_stats - non_overview_stats))
    PRUNED_CATEGORIES = {k: (sorted(v) if k != "Match overview" else keep_overview_only) for k, v in CATEGORIES.items()}
    ALL_STATS_UNIQUE = {s for lst in PRUNED_CATEGORIES.values() for s in lst}

    for col in ["home_team_stat", "away_team_stat"]:
        master_df[col] = (
            master_df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

    home_part = master_df[[
        "country", "tournament", "season", "week", "game_id",
        "stat_name", "home_team", "away_team", "home_team_stat"
    ]].rename(columns={"home_team":"team","away_team":"opponent","home_team_stat":"value"})
    home_part["side"] = "home"

    away_part = master_df[[
        "country", "tournament", "season", "week", "game_id",
        "stat_name", "home_team", "away_team", "away_team_stat"
    ]].rename(columns={"away_team":"team","home_team":"opponent","away_team_stat":"value"})
    away_part["side"] = "away"

    team_stats_long = pd.concat([home_part, away_part], ignore_index=True)
    team_stats_long = team_stats_long[team_stats_long["stat_name"].isin(ALL_STATS_UNIQUE)]

    df = team_stats_long.copy()
    if country:
        df = df[df["country"] == country]
    if league:
        df = df[df["tournament"] == league]
    if season:
        df = df[df["season"] == season]

    df = df[df["team"].notna() & df["game_id"].notna() & df["stat_name"].notna()]

    all_teams = df["team"].unique().tolist()
    if team not in all_teams:
        st.warning(f"No data available yet for {team} in {season} {league}.")
        return

    matches_per_team = df.groupby("team")["game_id"].nunique().rename("matches")

    features_wide = (
        df.pivot_table(index="team", columns="stat_name", values="value", aggfunc="mean")
        .sort_index()
    )
    features_wide = features_wide.loc[:, ~features_wide.columns.duplicated()]

    category_order = []
    for cat, stats in PRUNED_CATEGORIES.items():
        for s in stats:
            if s in features_wide.columns:
                category_order.append(s)
    features_wide = features_wide.reindex(columns=category_order)

    features = features_wide.merge(matches_per_team, left_index=True, right_index=True)

    feats_df = features.reset_index()
    all_cols = [c for c in feats_df.columns if c not in ["team", "matches"]]
    if len(all_cols) < 1:
        st.warning("Not enough features to compute similarity.")
        return

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    X_scaled = pipe.fit_transform(feats_df[all_cols])

    D = pairwise_distances(X_scaled, metric="cosine")
    S = 1.0 - D

    teams_list = feats_df["team"].astype(str).tolist()
    team_to_idx = {t: i for i, t in enumerate(teams_list)}

    if team not in team_to_idx:
        st.warning(f"Selected team '{team}' is not in the current filter.")
        return

    base_idx = team_to_idx[team]
    sims = S[base_idx]

    order = np.argsort(-sims)
    ranked = [{"similar_team": teams_list[j], "similarity": float(sims[j])}
              for j in order if j != base_idx]
    ranked_df = pd.DataFrame(ranked)

    if ranked_df.empty:
        st.warning("No other teams available for comparison.")
        return

    y_labels = ranked_df["similar_team"].tolist()
    x_vals = ranked_df["similarity"].values.astype(float)

    cmap = cm.get_cmap("RdBu_r")
    min_val = float(np.min([x_vals.min(), 0.0]))
    max_val = float(np.max([x_vals.max(), 0.0]))

    if np.isclose(min_val, max_val):
        rank_vals = np.linspace(1, -1, len(x_vals))
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        bar_colors = cmap(norm(rank_vals))
    else:
        norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=0.0, vmax=max_val)
        bar_colors = cmap(norm(x_vals))

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(y_labels, x_vals, color=bar_colors, edgecolor="k", alpha=0.95)
    ax.invert_yaxis()

    ax.axvline(0, color="gray", linewidth=1.0)

    ax.set_xlabel("Cosine similarity (-1 to 1)", labelpad=20)
    ax.set_title(
        f"{season} {league}\nSimilarity Ranking Relative to {team}\n(up to Week {max_week})",
        fontsize=16,
        fontweight="bold",
        pad=40
    )

    lim = max(abs(min_val), abs(max_val))
    ax.set_xlim(-lim - 0.05, lim + 0.05)

    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig)