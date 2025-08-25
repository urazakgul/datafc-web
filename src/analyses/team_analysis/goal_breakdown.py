import streamlit as st
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

def run(team: str, country: str, league: str, season: str):
    match_df, shots_df = require_session_data("match_data", "shots_data")

    match_df = filter_matches_by_status(match_df, "Ended")
    match_df = match_df[["season", "week", "game_id", "home_team", "away_team"]]

    max_week = match_df["week"].max()

    shots_df = shots_df.merge(
        match_df,
        on=["season", "week", "game_id"],
        how="left"
    )

    shots_df["team_name"] = shots_df.apply(
        lambda row: row["home_team"] if row["is_home"] else row["away_team"], axis=1
    )
    shots_df["is_goal"] = shots_df["shot_type"].apply(lambda x: 1 if x == "goal" else 0)
    shots_df["situation"] = shots_df["situation"].str.capitalize()
    shots_df["body_part"] = shots_df["body_part"].str.capitalize() if "body_part" in shots_df.columns else None
    shots_df["is_home"] = shots_df["is_home"].apply(lambda x: "Home" if x else "Away")

    shots_df_goals = shots_df[
        (shots_df["is_goal"] == 1) & (shots_df["goal_type"] != "own")
    ]

    team_goals = shots_df_goals[shots_df_goals["team_name"] == team]
    if team_goals.empty:
        st.warning(f"No data available yet for {team} in {season} {league}.")
        return

    breakdown_options = {
        "Situation": "situation",
        "Body Part": "body_part",
        "Time Interval": "time_interval",
        "Goal Mouth Location": "goal_mouth_location",
        "Player Position": "player_position",
        "Home/Away": "is_home"
    }

    selected_breakdown = st.selectbox(
        "Breakdown by:",
        list(breakdown_options.keys()),
        index=None,
        placeholder="Please select a breakdown category"
    )

    def _assign_time_interval(row):
        if row["time"] <= 45:
            if row.get("added_time", 0) == 0:
                if row["time"] <= 15:
                    return "1-15"
                elif row["time"] <= 30:
                    return "15-30"
                else:
                    return "30-45"
            else:
                return "45+"
        elif row["time"] > 45:
            if row.get("added_time", 0) == 0:
                if row["time"] <= 60:
                    return "45-60"
                elif row["time"] <= 75:
                    return "60-75"
                else:
                    return "75-90"
            else:
                return "90+"

    def _get_breakdown_df(df, group_col):
        if group_col == "time_interval":
            if "time_interval" not in df.columns:
                df = df.copy()
                df["time_interval"] = df.apply(_assign_time_interval, axis=1)
        pivot = (
            df
            .groupby(["team_name", group_col])
            .size()
            .unstack(fill_value=0)
        )
        team_total_goals = pivot.sum(axis=1)
        breakdown_df = (pivot.T / team_total_goals).T * 100
        breakdown_df = breakdown_df.round(1)
        breakdown_df.reset_index(inplace=True)
        return breakdown_df

    if selected_breakdown:
        group_col = breakdown_options[selected_breakdown]
        breakdown_df = _get_breakdown_df(shots_df_goals, group_col)

        team_row = breakdown_df[breakdown_df["team_name"] == team].copy()
        team_row = team_row.drop(columns=["team_name"])
        team_row = team_row.T
        team_row.columns = ["value"]
        if group_col == "time_interval":
            TIME_INTERVAL_ORDER = ["1-15", "15-30", "30-45", "45+", "45-60", "60-75", "75-90", "90+"]
            team_row = team_row.reindex(TIME_INTERVAL_ORDER).fillna(0)
        else:
            team_row = team_row.sort_values("value", ascending=False)

        bar_count = len(team_row)
        height = max(5, bar_count * 0.6)

        fig, ax = plt.subplots(figsize=(8, height))

        colors = plt.get_cmap("coolwarm_r")(plt.Normalize(team_row["value"].min(), team_row["value"].max())(team_row["value"]))

        bars = ax.bar(
            team_row.index,
            team_row["value"],
            color=colors,
            align="center"
        )
        ax.set_title(
            f"{season} {league}\n{selected_breakdown} Breakdown for {team}\n(up to Week {max_week})",
            fontsize=13,
            fontweight="bold",
            pad=30
        )
        ax.set_ylim(0, 100)
        ax.set_xlabel("")
        ax.set_ylabel("% of Goals")
        plt.tight_layout()

        for i, (category, v) in enumerate(zip(team_row.index, team_row["value"])):
            ax.text(i, v + 0.8, f"{v:.1f}", color="black", ha="center", fontsize=10)

        ax.set_xticks(range(len(team_row.index)))
        if selected_breakdown in ["Situation", "Goal Mouth Location"]:
            ax.set_xticklabels(team_row.index, rotation=30, ha="right")
        else:
            ax.set_xticklabels(team_row.index, rotation=0, ha="center")

        ax.grid(True, linestyle="--", alpha=0.7)

        st.pyplot(fig)