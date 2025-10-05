import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import require_session_data

from src.analyses.prediction.models.dixon_coles_model import (
    solve_parameters as dc_solve,
    dixon_coles_simulate_match as dc_sim,
)
from src.analyses.prediction.models.bivariate_poisson_model import (
    solve_bivariate_parameters as bp_solve,
    bivariate_poisson_simulate_match as bp_sim,
)
from src.analyses.prediction.models.skellam_distribution_model import (
    solve_skellam_parameters as sk_solve,
    skellam_goal_diff_probs as sk_diff,
)

@st.cache_data(show_spinner=False)
def _dc_solve_cached(df, weights):
    return dc_solve(df, weights=weights)

@st.cache_data(show_spinner=False)
def _bp_solve_cached(df):
    return bp_solve(df)

@st.cache_data(show_spinner=False)
def _sk_solve_cached(df):
    return sk_solve(df)

@st.cache_data(show_spinner=False)
def _dc_sim_cached(params, home, away, max_goals=10):
    return dc_sim(params, home, away, max_goals)

@st.cache_data(show_spinner=False)
def _bp_sim_cached(params, home, away, max_goals=10):
    return bp_sim(params, home, away, max_goals)

@st.cache_data(show_spinner=False)
def _sk_diff_cached(params, home, away, max_goals=10):
    return sk_diff(params, home, away, max_goals)

def _matrix_to_triple(mat: np.ndarray):
    if mat is None or mat.size == 0 or not np.any(mat):
        return None
    p_home = float(np.sum(np.tril(mat, -1)))
    p_draw = float(np.sum(np.diag(mat)))
    p_away = float(np.sum(np.triu(mat, 1)))
    s = p_home + p_draw + p_away
    if s <= 0:
        return None
    return (p_home / s, p_draw / s, p_away / s)

def _skellam_to_triple(diffs: np.ndarray, probs: np.ndarray):
    if diffs is None or probs is None or probs.size == 0 or not np.any(probs):
        return None
    p_home = float(probs[diffs > 0].sum())
    p_draw = float(probs[diffs == 0].sum())
    p_away = float(probs[diffs < 0].sum())
    s = p_home + p_draw + p_away
    if s <= 0:
        return None
    return (p_home / s, p_draw / s, p_away / s)

def _weighted_mean(triples, weights):
    w = np.array(weights, dtype=float)
    if np.all(w == 0):
        w = np.ones_like(w)
    w = w / w.sum()
    arr = np.array(triples, dtype=float)
    p = (arr * w[:, None]).sum(axis=0)
    p = p / p.sum()
    return tuple(p.tolist())

def _plot_ensemble_rows(rows, season, league, title, subtitle=None):
    if not rows:
        return
    df = pd.DataFrame(rows).reset_index(drop=True)

    n = len(df)

    per_bar_in = 0.50
    top_room_in = 1.80
    bottom_room_in_min = 0.60
    min_h_in = 4.5
    max_h_in = 18.0
    height_in = min(max_h_in, max(min_h_in, n * per_bar_in + top_room_in + bottom_room_in_min))

    bar_h = 0.65 if n <= 12 else 0.55 if n <= 20 else 0.45

    title_pad = int(np.clip(70 - 1.5 * n, 35, 70))

    legend_y = float(np.clip(1.18 - 0.005 * n, 1.04, 1.18))

    bottom_margin = float(np.clip(0.20 - 0.004 * n, 0.12, 0.20))

    subtitle_y = bottom_margin * (0.65 if n <= 10 else 0.60 if n <= 18 else 0.55)

    y = np.arange(n)
    home = (df["p_home"] * 100).values
    draw = (df["p_draw"] * 100).values
    away = (df["p_away"] * 100).values

    fig, ax = plt.subplots(figsize=(12, height_in))

    fig.subplots_adjust(top=0.78, bottom=bottom_margin, left=0.26, right=0.74)

    c_home = "#2E86AB"
    c_draw = "#F5B041"
    c_away = "#C0392B"

    ax.barh(y, home, label="Home", color=c_home, height=bar_h)
    ax.barh(y, draw, left=home, label="Draw", color=c_draw, height=bar_h)
    ax.barh(y, away, left=home + draw, label="Away", color=c_away, height=bar_h)

    ax.set_yticks(y)
    ax.set_yticklabels(df["home_team"])
    ax.invert_yaxis()
    ax.tick_params(axis="y", which="both", pad=8)
    for t in ax.get_yticklabels():
        t.set_horizontalalignment("right")

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    ax2.set_yticklabels(df["away_team"])
    ax2.tick_params(axis="y", which="both", pad=8, right=False, labelright=True)
    for t in ax2.get_yticklabels():
        t.set_horizontalalignment("left")
    ax2.grid(False)

    ax.set_xlim(0, 100)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=title_pad)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=3,
        frameon=False
    )

    if subtitle:
        fig.text(
            0.5, subtitle_y,
            subtitle,
            fontsize=9,
            fontstyle="italic",
            color="gray",
            ha="center", va="center"
        )

    ax.grid(False)

    for yi, (ph, pd_, pa) in enumerate(zip(home, draw, away)):
        if ph > 6:
            ax.text(ph / 2, yi, f"{ph:.0f}%", va="center", ha="center",
                    color="white", fontsize=9, fontweight="bold")
        if pd_ > 6:
            ax.text(ph + pd_ / 2, yi, f"{pd_:.0f}%", va="center", ha="center",
                    color="black", fontsize=9, fontweight="bold")
        if pa > 6:
            ax.text(ph + pd_ + pa / 2, yi, f"{pa:.0f}%", va="center", ha="center",
                    color="white", fontsize=9, fontweight="bold")

    st.pyplot(fig)
    plt.close(fig)

def run(country: str, league: str, season: str):
    match_df, historical_df = require_session_data("match_data", "tff_historical_matches")

    _match = match_df.copy()
    _match["status_norm"] = _match["status"].fillna("").str.strip().str.lower()
    priority_map = {"ended": 2, "not started": 1, "postponed": 0}
    _match["status_priority"] = _match["status_norm"].map(priority_map).fillna(-1).astype(int)
    keep_idx = (
        _match.groupby(["season", "week", "home_team", "away_team"], dropna=False)["status_priority"].idxmax()
    )
    match_df = (
        _match.loc[keep_idx]
        .drop(columns=["status_priority"])
        .reset_index(drop=True)
    )

    all_seasons = historical_df["season"].dropna().unique()
    selected_seasons = st.multiselect(
        "Select season(s) to include:",
        options=all_seasons,
        default=list(all_seasons),
        key="ensemble_seasons"
    )
    if not selected_seasons:
        st.warning("Please select at least one season.")
        return

    include_postponed = st.checkbox(
        "Include postponed matches",
        value=False,
        help="Turn on to also show postponed matches from earlier weeks. They will appear under their original weeks.",
        key="include_postponed_ensemble"
    )

    st.markdown("#### Ensemble Weights")
    st.caption("Adjust each model's influence. Values are normalized to sum to 1.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        w_dc = st.slider("Dixon-Coles", 0.0, 1.0, 0.25, 0.01, key="w_dc")
    with col2:
        w_dc_decay = st.slider("Dixon-Coles (Time Decay)", 0.0, 1.0, 0.25, 0.01, key="w_dc_decay")
    with col3:
        w_bp = st.slider("Bivariate Poisson", 0.0, 1.0, 0.25, 0.01, key="w_bp")
    with col4:
        w_sk = st.slider("Skellam Distribution", 0.0, 1.0, 0.25, 0.01, key="w_sk")

    _total = w_dc + w_dc_decay + w_bp + w_sk
    if abs(_total - 1.0) > 0.01:
        st.warning(f"The total weight should sum to 1. Currently: {_total:.2f}")
        run_button_disabled = True
    else:
        run_button_disabled = False
    if _total == 0:
        w_dc = w_dc_decay = w_bp = w_sk = 0.25
    else:
        w_dc, w_dc_decay, w_bp, w_sk = (
            w_dc/_total, w_dc_decay/_total, w_bp/_total, w_sk/_total
        )

    weights_tuple = (w_dc, w_dc_decay, w_bp, w_sk)

    filtered_historical = historical_df[historical_df["season"].isin(selected_seasons)].copy()

    season_df = match_df[match_df["season"] == season].copy()
    if season_df.empty or season_df["week"].isna().all():
        st.warning("No matches found for the selected season.")
        return

    status_norm = season_df["status"].fillna("").astype(str).str.strip().str.lower()
    last_week = season_df["week"].max()

    postponed_df = season_df[status_norm == "postponed"].copy() if include_postponed else season_df.iloc[0:0].copy()
    upcoming_df = season_df[(season_df["week"] == last_week) & (status_norm != "postponed")].copy()

    target_matches = pd.concat([postponed_df, upcoming_df], ignore_index=True)
    if target_matches.empty:
        st.warning("No matches found to display for the current settings.")
        return

    week_teams = pd.unique(target_matches[["home_team", "away_team"]].values.ravel())
    filtered_historical = filtered_historical[
        filtered_historical["home_team"].isin(week_teams) &
        filtered_historical["away_team"].isin(week_teams)
    ]
    filtered_historical = filtered_historical[
        ~((filtered_historical["season"] == season) & (filtered_historical["week"] >= last_week))
    ]

    filtered_historical["home_score"] = pd.to_numeric(filtered_historical["home_score"], errors="coerce")
    filtered_historical["away_score"] = pd.to_numeric(filtered_historical["away_score"], errors="coerce")
    filtered_historical = filtered_historical.dropna(subset=["home_score", "away_score"]).rename(
        columns={"home_score": "home_team_goals", "away_score": "away_team_goals"}
    )
    if filtered_historical.empty:
        st.warning("No historical data available after filtering.")
        return

    filtered_historical["home_team_goals"] = filtered_historical["home_team_goals"].astype(int)
    filtered_historical["away_team_goals"] = filtered_historical["away_team_goals"].astype(int)

    dc_classic_weights = np.ones(len(filtered_historical))

    filtered_historical["date"] = pd.to_datetime(filtered_historical["date"], errors="coerce")
    if filtered_historical["date"].notna().any():
        xi = 0.0018
        max_date = filtered_historical["date"].max()
        days_diff = (max_date - filtered_historical["date"]).dt.days.fillna(0).values
        dc_decay_weights = np.exp(-xi * days_diff)
        dc_decay_weights = dc_decay_weights / np.mean(dc_decay_weights)
    else:
        dc_decay_weights = np.ones(len(filtered_historical))

    if st.button("Run Model for All Matches", key="ensemble_calc_button_all", disabled=run_button_disabled):
        with st.spinner("Running the ensemble for all matches... It might take a bit longer while the system is being set up."):
            dc_params = _dc_solve_cached(filtered_historical, dc_classic_weights)
            dc_decay_params = _dc_solve_cached(filtered_historical, dc_decay_weights)
            bp_params = _bp_solve_cached(filtered_historical)
            sk_params = _sk_solve_cached(filtered_historical)

        def _predict_one(home, away):
            t_dc = _matrix_to_triple(_dc_sim_cached(dc_params, home, away, max_goals=10))
            t_dc_decay = _matrix_to_triple(_dc_sim_cached(dc_decay_params, home, away, max_goals=10))
            t_bp = _matrix_to_triple(_bp_sim_cached(bp_params, home, away, max_goals=10))
            diffs, probs = _sk_diff_cached(sk_params, home, away, max_goals=10)
            t_sk = _skellam_to_triple(diffs, probs)

            triples = [t for t in (t_dc, t_dc_decay, t_bp, t_sk) if t is not None]
            ws = []
            if t_dc is not None: ws.append(weights_tuple[0])
            if t_dc_decay is not None: ws.append(weights_tuple[1])
            if t_bp is not None: ws.append(weights_tuple[2])
            if t_sk is not None: ws.append(weights_tuple[3])

            if not triples:
                return None
            return _weighted_mean(triples, ws)

        def _run_block(df, title, week_value):
            if df.empty:
                return
            st.markdown(title)
            rows = []
            for _, r in df.iterrows():
                res = _predict_one(r["home_team"], r["away_team"])
                if res is None:
                    continue
                p_home, p_draw, p_away = res
                rows.append({
                    "week": int(week_value),
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "p_home": p_home,
                    "p_draw": p_draw,
                    "p_away": p_away,
                })

            if rows:
                subtitle = (
                    f"Model Weights -- "
                    f"Dixon-Coles: {w_dc:.2f}, Dixon-Coles (Time Decay): {w_dc_decay:.2f}, "
                    f"Bivariate Poisson: {w_bp:.2f}, Skellam Dist.: {w_sk:.2f}"
                )

                _plot_ensemble_rows(
                    rows, season, league,
                    f"{season} {league} – Week {int(week_value)} Ensemble Match Probabilities (%)",
                    subtitle
                )

        if include_postponed and not postponed_df.empty:
            st.markdown("## Postponed Matches")
            for wk in sorted([int(w) for w in postponed_df["week"].dropna().unique()]):
                wk_group = postponed_df[postponed_df["week"] == wk]
                _run_block(
                    wk_group,
                    f"### {season} {league} - Week {wk} Predictions (Postponed) (Ensemble)",
                    wk
                )

        if not upcoming_df.empty:
            st.markdown("## Upcoming Week")
            _run_block(
                upcoming_df,
                f"### {season} {league} - Week {int(last_week)} Predictions (Ensemble)",
                last_week
            )