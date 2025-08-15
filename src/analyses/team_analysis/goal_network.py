import streamlit as st
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
import itertools
from matplotlib.colors import to_hex

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

EVENT_COLORS = {
    "pass": "#0074D9",
    "goal": "#FF4136",
    "free-kick": "#2ECC40",
    "clearance": "#B10DC9",
    "ball-movement": "#FF851B",
    "corner": "#FFDC00",
    "post": "#FF69B4",
    "save": "#7FDBFF",
    "miss": "#AAAAAA",
}

def _fill_by_id(df, col):
    df[col] = df.groupby("id")[col].transform(lambda x: x.ffill().bfill())
    return df

def _merge_match_data(match_data_df, shots_data_df):
    filtered_shots = shots_data_df[shots_data_df["shot_type"] == "goal"][
        ["tournament", "season", "week", "game_id", "player_name", "is_home", "goal_type", "xg"]
    ]
    merged_df = match_data_df.merge(filtered_shots, on=["tournament", "season", "week", "game_id"])
    return merged_df[~merged_df["goal_type"].isin(["penalty", "own"])]

def _get_dynamic_color_map(event_types, static_map=EVENT_COLORS):
    used_colors = set(to_hex(c) for c in static_map.values())
    palette = itertools.cycle(plt.get_cmap("tab20").colors)
    dynamic_map = {k: to_hex(v) for k, v in static_map.items()}

    for ev in sorted(event_types):
        if ev not in dynamic_map:
            candidate = to_hex(next(palette))
            while candidate in used_colors:
                candidate = to_hex(next(palette))
            dynamic_map[ev] = candidate
            used_colors.add(candidate)
    return dynamic_map

def _add_event_legend(ax, event_colors):
    keys = sorted(event_colors.keys())
    handles = [
        plt.Line2D([0], [0], marker="o", color=event_colors[k], markersize=7, linestyle="None")
        for k in keys
    ]
    labels = [k.capitalize() for k in keys]
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=False,
        ncol=3,
        fontsize=8,
    )

def _plot_goal_network(data, pitch, ax, line_color, title, event_colors=EVENT_COLORS, order_cols=None):
    all_events = data["event_type"].dropna().unique()
    event_colors = _get_dynamic_color_map(all_events, static_map=event_colors)

    if order_cols:
        data = data.sort_values(order_cols)

    for _, row in data.iterrows():
        color = event_colors.get(row["event_type"], "#888888")
        pitch.scatter(
            row["player_x"],
            row["player_y"],
            ax=ax,
            color=color,
            s=50,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
            zorder=2,
        )

    grouped = data.groupby("id", sort=False)
    for _, group in grouped:
        xs = group["player_x"].to_numpy()
        ys = group["player_y"].to_numpy()
        if len(xs) > 1:
            pitch.lines(
                xs[:-1],
                ys[:-1],
                xs[1:],
                ys[1:],
                ax=ax,
                lw=1,
                color=line_color,
                alpha=0.2,
                zorder=1,
            )

    ax.set_title(title, fontsize=10, fontweight="bold", loc="center", pad=45)
    _add_event_legend(ax, event_colors)

def _flip_to_right_attacking_half(df):
    for _, g in df.groupby("id"):
        if (g["event_type"] == "goal").any():
            goal_row = g.loc[g["event_type"] == "goal"].iloc[0]
            if "goal_shot_x" in goal_row and goal_row["goal_shot_x"] != 100:
                df.loc[g.index, ["player_x", "player_y"]] = 100 - g[["player_x", "player_y"]]
    return df

def _render_chart(data, title, line_color="#1f77b4"):
    pitch = VerticalPitch(
        pitch_type="opta",
        corner_arcs=True,
        half=False,
        label=False,
        tick=False
    )
    fig, ax = pitch.draw(figsize=(7, 10))
    _plot_goal_network(
        data=data,
        pitch=pitch,
        ax=ax,
        line_color=line_color,
        title=title
    )
    st.pyplot(fig)

def run(team: str, country: str, league: str, season: str):
    match_df, shots_df, goal_networks_df = require_session_data("match_data", "shots_data", "goal_networks_data")

    info_box = st.empty()
    info_box.info("Two separate charts will be displayed below shortly.")

    match_df = filter_matches_by_status(match_df, "Ended")
    match_df = match_df[["tournament", "season", "week", "game_id", "home_team", "away_team"]]

    max_week = match_df["week"].max()

    match_shots_df = _merge_match_data(match_df, shots_df)

    goal_networks_data_scored_df = goal_networks_df.copy()
    goal_networks_data_conceded_df = goal_networks_df.copy()

    goal_networks_data_scored_df["team_name"] = None
    goal_networks_data_conceded_df["opponent_team_name"] = None

    for game_id in match_shots_df["game_id"].unique():
        match_data = match_shots_df[match_shots_df["game_id"] == game_id]
        for _, row in match_data.iterrows():
            team_name = row["home_team"] if row["is_home"] else row["away_team"]
            opponent_team_name = row["away_team"] if row["is_home"] else row["home_team"]

            goal_networks_data_scored_df.loc[
                (goal_networks_data_scored_df["game_id"] == game_id) &
                (goal_networks_data_scored_df["player_name"] == row["player_name"]) &
                (goal_networks_data_scored_df["event_type"] == "goal"),
                "team_name"
            ] = team_name

            goal_networks_data_conceded_df.loc[
                (goal_networks_data_conceded_df["game_id"] == game_id) &
                (goal_networks_data_conceded_df["player_name"] == row["player_name"]) &
                (goal_networks_data_conceded_df["event_type"] == "goal"),
                "opponent_team_name"
            ] = opponent_team_name

    goal_networks_scored_df = _fill_by_id(goal_networks_data_scored_df, "team_name")
    goal_networks_conceded_df = _fill_by_id(goal_networks_data_conceded_df, "opponent_team_name")

    goal_networks_scored_df = _flip_to_right_attacking_half(goal_networks_scored_df)
    goal_networks_conceded_df = _flip_to_right_attacking_half(goal_networks_conceded_df)

    goal_networks_scored_df = goal_networks_scored_df.merge(match_df, on=["tournament", "season", "week", "game_id"])
    goal_networks_conceded_df = goal_networks_conceded_df.merge(match_df, on=["tournament", "season", "week", "game_id"])

    side_data_for = goal_networks_scored_df[goal_networks_scored_df["team_name"] == team]
    side_data_against = goal_networks_conceded_df[goal_networks_conceded_df["opponent_team_name"] == team]

    if side_data_for.empty and side_data_against.empty:
        info_box.warning(f"No data available yet for {team} in {season} {league}.")

    elif not side_data_for.empty and not side_data_against.empty:
        _render_chart(
            side_data_for,
            f"{season} {league}\nGoal Networks (Scored) for {team}\n(up to Week {max_week})"
        )
        _render_chart(
            side_data_against,
            f"{season} {league}\nGoal Networks (Conceded) for {team}\n(up to Week {max_week})"
        )
        info_box.success(
            f"Both 'Scored' and 'Conceded' goal networks for {team} are displayed below."
        )

    else:
        if not side_data_for.empty:
            _render_chart(
                side_data_for,
                f"{season} {league}\nGoal Networks (Scored) for {team}\n(up to Week {max_week})"
            )
            info_box.success(f"Only 'Scored' goal networks for {team} are available. No data for 'Conceded'.")
        else:
            _render_chart(
                side_data_against,
                f"{season} {league}\nGoal Networks (Conceded) for {team}\n(up to Week {max_week})"
            )
            info_box.success(f"Only 'Conceded' goal networks for {team} are available. No data for 'Scored'.")