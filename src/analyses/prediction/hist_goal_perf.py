import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import require_session_data

def _goals_for_against_counts(df_: pd.DataFrame, team_: str):
    dft = df_[(df_["home_team"] == team_) | (df_["away_team"] == team_)].copy()
    if dft.empty:
        return pd.Series(dtype=int), pd.Series(dtype=int), 0

    gf = np.where(dft["home_team"] == team_, dft["home_score"], dft["away_score"])
    ga = np.where(dft["home_team"] == team_, dft["away_score"], dft["home_score"])
    gf = pd.Series(gf).dropna().astype(int)
    ga = pd.Series(ga).dropna().astype(int)

    gf_max = int(gf.max()) if not gf.empty else 0
    ga_max = int(ga.max()) if not ga.empty else 0
    max_goal_ = max(gf_max, ga_max)

    idx_ = pd.Index(range(0, max_goal_ + 1), name="goals")
    gf_counts = gf.value_counts().reindex(idx_, fill_value=0).sort_index()
    ga_counts = ga.value_counts().reindex(idx_, fill_value=0).sort_index()
    matches_used = int(len(dft))
    return gf_counts, ga_counts, matches_used

def _add_grouped_pct_labels(ax, x, pct_values, offset, dy=0.8):
    for xi, h in zip(x, pct_values):
        if h >= 0.5:
            ax.text(
                xi + offset,
                h + dy,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=False
            )

def _set_percentage_axis(ax):
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

def _team_cum_scored_conceded_diff(df: pd.DataFrame, team: str) -> pd.DataFrame:
    team_df = df[(df["home_team"] == team) | (df["away_team"] == team)].copy()

    team_df["goals_for"] = np.where(team_df["home_team"] == team,
                                    team_df["home_score"], team_df["away_score"])
    team_df["goals_against"] = np.where(team_df["home_team"] == team,
                                        team_df["away_score"], team_df["home_score"])

    team_df["goals_for"] = pd.to_numeric(team_df["goals_for"], errors="coerce")
    team_df["goals_against"] = pd.to_numeric(team_df["goals_against"], errors="coerce")
    team_df["date"] = pd.to_datetime(team_df["date"], errors="coerce")

    team_df = team_df.dropna(subset=["goals_for", "goals_against", "date"]).sort_values("date")
    team_df["match_idx"] = np.arange(1, len(team_df) + 1)

    team_df["cum_for"] = team_df["goals_for"].cumsum()
    team_df["cum_against"] = team_df["goals_against"].cumsum()
    team_df["cum_diff"] = team_df["cum_for"] - team_df["cum_against"]

    return team_df[["match_idx", "cum_for", "cum_against", "cum_diff"]]

def run(country: str, league: str, season: str):
    match_df, historical_df = require_session_data("match_data", "tff_historical_matches")

    mode_map = {
        "Scoreline distribution": "distribution",
        "Season progress (cumulative goals & goal difference)": "cumulative",
    }
    mode_choice = st.radio(
        "What would you like to analyze?",
        list(mode_map.keys()),
        horizontal=True,
        key="historical_mode"
    )
    mode = mode_map[mode_choice]

    all_seasons = historical_df['season'].dropna().unique()
    selected_seasons = st.multiselect(
        "Select season(s) to use in the model:",
        options=all_seasons,
        default=list(all_seasons),
        key="historical_seasons"
    )
    if not selected_seasons:
        st.warning("Please select at least one season.")
        return

    filtered_historical = historical_df[historical_df["season"].isin(selected_seasons)]
    match_df_filtered = match_df[match_df['season'] == season]

    last_week = match_df_filtered['week'].max()
    last_week_matches = match_df_filtered[match_df_filtered['week'] == last_week]

    match_options = [
        f"{row['home_team']} - {row['away_team']}"
        for _, row in last_week_matches.iterrows()
    ]
    match_options_unique = list(dict.fromkeys(match_options))

    selected_match = st.selectbox(
        f"Select a match from week {last_week} ({len(match_options_unique)} matches):",
        match_options_unique,
        index=None,
        placeholder="Select a match",
        key="historical_selected_match"
    )

    if not selected_match:
        return

    match_row = last_week_matches.iloc[match_options.index(selected_match)]
    home_team = match_row["home_team"]
    away_team = match_row["away_team"]

    filtered_historical = filtered_historical[
        ~(
            (filtered_historical['season'] == season) &
            (filtered_historical['week'] >= last_week)
        )
    ]

    filtered_historical["home_score"] = pd.to_numeric(filtered_historical["home_score"], errors="coerce")
    filtered_historical["away_score"] = pd.to_numeric(filtered_historical["away_score"], errors="coerce")
    filtered_historical = filtered_historical.dropna(subset=["home_score", "away_score"])

    seasons_str = ", ".join(selected_seasons)

    if mode == "distribution":
        home_for, home_against, total_home = _goals_for_against_counts(filtered_historical, home_team)
        away_for, away_against, total_away = _goals_for_against_counts(filtered_historical, away_team)

        if total_home == 0 or total_away == 0:
            if total_home == 0:
                st.info(f"No historical goal data found for **{home_team}** with current filters.")
            if total_away == 0:
                st.info(f"No historical goal data found for **{away_team}** with current filters.")
            st.stop()

        max_goal = max(home_for.index.max(), away_for.index.max())
        common_idx = pd.Index(range(0, max_goal + 1), name="goals")
        home_for, home_against = home_for.reindex(common_idx, fill_value=0), home_against.reindex(common_idx, fill_value=0)
        away_for, away_against = away_for.reindex(common_idx, fill_value=0), away_against.reindex(common_idx, fill_value=0)

        home_for_pct = (home_for / home_for.sum()) * 100
        home_against_pct = (home_against / home_against.sum()) * 100
        away_for_pct = (away_for / away_for.sum()) * 100
        away_against_pct = (away_against / away_against.sum()) * 100

        scored_color = "#7561f4"
        conceded_color = "#e75168"

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        width = 0.4
        x = np.arange(len(common_idx))

        axes[0].bar(x - width/2, home_for_pct.values,     width=width, color=scored_color,   alpha=0.85, label="Scored")
        axes[0].bar(x + width/2, home_against_pct.values, width=width, color=conceded_color, alpha=0.85, label="Conceded")
        _add_grouped_pct_labels(axes[0], x, home_for_pct.values,     offset=-width/2)
        _add_grouped_pct_labels(axes[0], x, home_against_pct.values, offset=+width/2)
        axes[0].set_title(f"{home_team}", pad=20)
        axes[0].grid(True, linestyle="--", alpha=0.7)
        _set_percentage_axis(axes[0])

        axes[1].bar(x - width/2, away_for_pct.values,     width=width, color=scored_color,   alpha=0.85)
        axes[1].bar(x + width/2, away_against_pct.values, width=width, color=conceded_color, alpha=0.85)
        _add_grouped_pct_labels(axes[1], x, away_for_pct.values,     offset=-width/2)
        _add_grouped_pct_labels(axes[1], x, away_against_pct.values, offset=+width/2)
        axes[1].set_title(f"{away_team}", pad=20)
        axes[1].set_xlabel("Goals")
        axes[1].grid(True, linestyle="--", alpha=0.7)
        _set_percentage_axis(axes[1])

        axes[0].set_xticks(x); axes[0].set_xticklabels(common_idx.tolist())
        axes[1].set_xticks(x); axes[1].set_xticklabels(common_idx.tolist())
        fig.text(-0.03, 0.5, "Goal Count Distribution (%)", va="center", rotation="vertical")

        fig.legend(["Scored", "Conceded"], loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=False)
        fig.suptitle(
            f"{seasons_str} Seasons\n{home_team} vs. {away_team} – Goals Scored & Conceded Distribution",
            fontsize=14,
            fontweight="bold",
            y=1.07
        )
        fig.text(
            0.5, 0.98,
            f"Matches Played – {home_team}: {total_home} | {away_team}: {total_away}",
            ha="center",
            fontsize=12,
            color="gray"
        )

        plt.tight_layout()
        st.pyplot(fig)

    elif mode == "cumulative":
        home_ts = _team_cum_scored_conceded_diff(filtered_historical, home_team)
        away_ts = _team_cum_scored_conceded_diff(filtered_historical, away_team)

        if home_ts.empty or away_ts.empty:
            if home_ts.empty:
                st.info(f"No historical time series could be generated for **{home_team}** with current filters.")
            if away_ts.empty:
                st.info(f"No historical time series could be generated for **{away_team}** with current filters.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=False, sharey=True)

        ax = axes[0]
        ax.plot(home_ts["match_idx"], home_ts["cum_for"], linewidth=2, alpha=0.35, color="green", label="Cumulative Scored")
        ax.plot(home_ts["match_idx"], home_ts["cum_against"], linewidth=2, alpha=0.35, color="red", label="Cumulative Conceded")
        ax.plot(home_ts["match_idx"], home_ts["cum_diff"], linewidth=3, color="black", label="Cumulative Goal Difference", zorder=5)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_title(f"{home_team}", pad=20)
        ax.grid(True, linestyle="--", alpha=0.7)

        ax = axes[1]
        ax.plot(away_ts["match_idx"], away_ts["cum_for"], linewidth=2, alpha=0.35, color="green", label="Cumulative Scored")
        ax.plot(away_ts["match_idx"], away_ts["cum_against"], linewidth=2, alpha=0.35, color="red", label="Cumulative Conceded")
        ax.plot(away_ts["match_idx"], away_ts["cum_diff"], linewidth=3, color="black", label="Cumulative Goal Difference", zorder=5)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_title(f"{away_team}", pad=20)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlabel("Match Index (sorted by date)", labelpad=18)

        fig.text(-0.03, 0.5, "Cumulative Goals & Goal Difference", va="center", rotation="vertical")

        fig.legend(
            ["Cumulative Scored", "Cumulative Conceded", "Cumulative Goal Difference"],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            fontsize=12,
            ncol=3,
            frameon=False
        )

        fig.suptitle(
            f"{seasons_str} Seasons\n{home_team} vs. {away_team} – Cumulative Scored, Conceded & Difference",
            fontsize=14,
            fontweight="bold",
            y=1.02
        )
        plt.tight_layout()
        st.pyplot(fig)