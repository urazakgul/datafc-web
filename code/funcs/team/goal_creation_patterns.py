import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer, sort_turkish
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_goal_share_plot(team_goal_types_df, category, category_en, subcategory, last_round):
    heatmap_data = team_goal_types_df.pivot(
        index="team_name",
        columns=category_en,
        values="goal_share" if subcategory == "By Team Share" else "team_share"
    )
    if st.session_state["selected_league"] == "super_lig":
        heatmap_data = sort_turkish(heatmap_data.reset_index(), column="team_name").set_index("team_name")
    else:
        heatmap_data = heatmap_data.sort_index()
    if category_en == "player_position":
        existing_columns = [col for col in ["G", "D", "M", "F"] if col in heatmap_data.columns]
        heatmap_data = heatmap_data[existing_columns]
    elif category_en == "is_home":
        existing_columns = [col for col in ["Home", "Away"] if col in heatmap_data.columns]
        heatmap_data = heatmap_data[existing_columns]

    fig, ax = plt.subplots(figsize=(12,12))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        cbar=False,
        ax=ax
    )

    title_suffix = "(Per Team)" if subcategory == "By Team Share" else "(Across Team)"
    ax.set_title(
        f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Percentage of Goals Scored by {category} {title_suffix}",
        fontsize=12,
        fontweight="bold",
        pad=30
    )
    ax.set_xlabel("Goal Type", fontsize=12, labelpad=20)
    ax.set_ylabel("")

    plt.xticks(rotation=45 if category_en == "situation" else 0, fontsize=12)
    plt.tight_layout()
    add_footer(fig, y=-0.01, extra_text="Opponent's own goals excluded.")

    st.pyplot(fig)

def main(category, subcategory):
    try:

        match_data_df = get_data("match_data")
        shots_data_df = get_data("shots_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]
        match_data_df = match_data_df[["season", "week", "game_id", "home_team", "away_team"]]

        shots_data_df = shots_data_df[[
            "season", "week", "game_id", "player_name", "player_position", "is_home", "shot_type", "body_part", "goal_type",
            "situation", "goal_mouth_location", "player_coordinates_x", "player_coordinates_y", "time", "added_time"
        ]]

        shots_data_df = shots_data_df.merge(
            match_data_df,
            on=["season", "week", "game_id"],
            how="left"
        )

        shots_data_df["team_name"] = shots_data_df.apply(
            lambda row: row["home_team"] if row["is_home"] else row["away_team"], axis=1
        )

        shots_data_df["is_goal"] = shots_data_df["shot_type"].apply(lambda x: 1 if x == "goal" else 0)

        shot_maps_data_goals = shots_data_df[shots_data_df["is_goal"] == 1]
        shot_maps_data_goals = shot_maps_data_goals[shot_maps_data_goals['goal_type'] != "own"]

        shot_maps_data_goals["is_home"] = shot_maps_data_goals["is_home"].apply(lambda x: "Home" if x else "Away")

        shot_maps_data_goals["situation"] = shot_maps_data_goals["situation"].str.capitalize()

        if category == "Situation":
            category_en = "situation"
            if subcategory == "By Team Share":
                team_goal_types_df = (
                    shot_maps_data_goals
                    .groupby(["team_name", "situation"])
                    .size()
                    .reset_index(name="goal_count")
                    .merge(
                        shot_maps_data_goals.groupby("team_name")["is_goal"].sum().reset_index(name="total_goals"),
                        on="team_name",
                        how="left"
                    )
                    .assign(goal_share=lambda df: df["goal_count"] / df["total_goals"] * 100)
                    .pivot(index="team_name", columns="situation", values="goal_share")
                    .fillna(0)
                    .reset_index()
                    .melt(id_vars=["team_name"], var_name="situation", value_name="goal_share")
                )
            elif subcategory == "By Team Comparison Share":
                team_goal_types_df = (
                    shot_maps_data_goals
                    .groupby(["situation", "team_name"])
                    .size()
                    .reset_index(name="goal_count")
                    .merge(
                        shot_maps_data_goals.groupby("situation")["is_goal"].sum().reset_index(name="total_goals_by_situation"),
                        on="situation",
                        how="left"
                    )
                    .assign(team_share=lambda df: df["goal_count"] / df["total_goals_by_situation"] * 100)
                    .pivot(index="team_name", columns="situation", values="team_share")
                    .fillna(0)
                    .reset_index()
                    .melt(id_vars=["team_name"], var_name="situation", value_name="team_share")
                )

        elif category == "Body Part":
            category_en = "body_part"
            if subcategory == "By Team Share":
                team_goal_types_df = (
                    shot_maps_data_goals
                    .groupby(["team_name", "body_part"])
                    .size()
                    .reset_index(name="goal_count")
                    .merge(
                        shot_maps_data_goals.groupby("team_name")["is_goal"].sum().reset_index(name="total_goals"),
                        on="team_name",
                        how="left"
                    )
                    .assign(goal_share=lambda df: df["goal_count"] / df["total_goals"] * 100)
                    .pivot(index="team_name", columns="body_part", values="goal_share")
                    .fillna(0)
                    .reset_index()
                    .melt(id_vars=["team_name"], var_name="body_part", value_name="goal_share")
                )
            elif subcategory == "By Team Comparison Share":
                team_goal_types_df = (
                    shot_maps_data_goals
                    .groupby(["body_part", "team_name"])
                    .size()
                    .reset_index(name="goal_count")
                    .merge(
                        shot_maps_data_goals.groupby("body_part")["is_goal"].sum().reset_index(name="total_goals_by_body_part"),
                        on="body_part",
                        how="left"
                    )
                    .assign(team_share=lambda df: df["goal_count"] / df["total_goals_by_body_part"] * 100)
                    .pivot(index="team_name", columns="body_part", values="team_share")
                    .fillna(0)
                    .reset_index()
                    .melt(id_vars=["team_name"], var_name="body_part", value_name="team_share")
                )

        elif category == "Goal Mouth Location":
            category_en = "goal_mouth_location"
            if subcategory == "By Team Share":
                team_goal_types_df = (
                    shot_maps_data_goals
                    .groupby(["team_name", "goal_mouth_location"])
                    .size()
                    .reset_index(name="goal_count")
                    .merge(
                        shot_maps_data_goals.groupby("team_name")["is_goal"].sum().reset_index(name="total_goals"),
                        on="team_name",
                        how="left"
                    )
                    .assign(goal_share=lambda df: df["goal_count"] / df["total_goals"] * 100)
                    .pivot(index="team_name", columns="goal_mouth_location", values="goal_share")
                    .fillna(0)
                    .reset_index()
                    .melt(id_vars=["team_name"], var_name="goal_mouth_location", value_name="goal_share")
                )
            elif subcategory == "By Team Comparison Share":
                team_goal_types_df = (
                    shot_maps_data_goals
                    .groupby(["goal_mouth_location", "team_name"])
                    .size()
                    .reset_index(name="goal_count")
                    .merge(
                        shot_maps_data_goals.groupby("goal_mouth_location")["is_goal"].sum().reset_index(name="total_goals_by_location"),
                        on="goal_mouth_location",
                        how="left"
                    )
                    .assign(team_share=lambda df: df["goal_count"] / df["total_goals_by_location"] * 100)
                    .pivot(index="team_name", columns="goal_mouth_location", values="team_share")
                    .fillna(0)
                    .reset_index()
                    .melt(id_vars=["team_name"], var_name="goal_mouth_location", value_name="team_share")
                )

        elif category == "Time Interval":
            def assign_time_interval(row):
                if row["time"] <= 45:
                    if row["added_time"] == 0:
                        if row["time"] <= 15:
                            return "1-15"
                        elif row["time"] <= 30:
                            return "15-30"
                        else:
                            return "30-45"
                    else:
                        return "45+"
                elif row["time"] > 45:
                    if row["added_time"] == 0:
                        if row["time"] <= 60:
                            return "45-60"
                        elif row["time"] <= 75:
                            return "60-75"
                        else:
                            return "75-90"
                    else:
                        return "90+"

            shot_maps_data_goals["time_interval"] = shot_maps_data_goals.apply(assign_time_interval, axis=1)

            category_en = "time_interval"
            if subcategory == "By Team Share":
                team_goal_types_df = (
                    shot_maps_data_goals
                    .groupby(["team_name", "time_interval"])
                    .size()
                    .reset_index(name="goal_count")
                    .merge(
                        shot_maps_data_goals.groupby("team_name")["is_goal"].sum().reset_index(name="total_goals"),
                        on="team_name",
                        how="left"
                    )
                    .assign(goal_share=lambda df: df["goal_count"] / df["total_goals"] * 100)
                    .pivot(index="team_name", columns="time_interval", values="goal_share")
                    .fillna(0)
                    .reset_index()
                    .melt(id_vars=["team_name"], var_name="time_interval", value_name="goal_share")
                )
            elif subcategory == "By Team Comparison Share":
                team_goal_types_df = (
                    shot_maps_data_goals
                    .groupby(["time_interval", "team_name"])
                    .size()
                    .reset_index(name="goal_count")
                    .merge(
                        shot_maps_data_goals.groupby("time_interval")["is_goal"].sum().reset_index(name="total_goals_by_time"),
                        on="time_interval",
                        how="left"
                    )
                    .assign(team_share=lambda df: df["goal_count"] / df["total_goals_by_time"] * 100)
                    .pivot(index="team_name", columns="time_interval", values="team_share")
                    .fillna(0)
                    .reset_index()
                    .melt(id_vars=["team_name"], var_name="time_interval", value_name="team_share")
                )

        elif category == "Player Position":
            category_en = "player_position"
            if subcategory == "By Team Share":
                team_goal_types_df = (
                    shot_maps_data_goals
                    .groupby(["team_name", "player_position"])
                    .size()
                    .reset_index(name="goal_count")
                    .merge(
                        shot_maps_data_goals.groupby("team_name")["is_goal"].sum().reset_index(name="total_goals"),
                        on="team_name",
                        how="left"
                    )
                    .assign(goal_share=lambda df: df["goal_count"] / df["total_goals"] * 100)
                    .pivot(index="team_name", columns="player_position", values="goal_share")
                    .fillna(0)
                    .reset_index()
                    .melt(id_vars=["team_name"], var_name="player_position", value_name="goal_share")
                )
            elif subcategory == "By Team Comparison Share":
                team_goal_types_df = (
                    shot_maps_data_goals
                    .groupby(["player_position", "team_name"])
                    .size()
                    .reset_index(name="goal_count")
                    .merge(
                        shot_maps_data_goals.groupby("player_position")["is_goal"].sum().reset_index(name="total_goals_by_location"),
                        on="player_position",
                        how="left"
                    )
                    .assign(team_share=lambda df: df["goal_count"] / df["total_goals_by_location"] * 100)
                    .pivot(index="team_name", columns="player_position", values="team_share")
                    .fillna(0)
                    .reset_index()
                    .melt(id_vars=["team_name"], var_name="player_position", value_name="team_share")
                )

        elif category == "Home-Away":
            category_en = "is_home"
            if subcategory == "By Team Share":
                team_goal_types_df = (
                    shot_maps_data_goals
                    .groupby(["team_name", "is_home"])
                    .size()
                    .reset_index(name="goal_count")
                    .merge(
                        shot_maps_data_goals.groupby("team_name")["is_goal"].sum().reset_index(name="total_goals"),
                        on="team_name",
                        how="left"
                    )
                    .assign(goal_share=lambda df: df["goal_count"] / df["total_goals"] * 100)
                    .pivot(index="team_name", columns="is_home", values="goal_share")
                    .fillna(0)
                    .reset_index()
                    .melt(id_vars=["team_name"], var_name="is_home", value_name="goal_share")
                )
            elif subcategory == "By Team Comparison Share":
                team_goal_types_df = (
                    shot_maps_data_goals
                    .groupby(["is_home", "team_name"])
                    .size()
                    .reset_index(name="goal_count")
                    .merge(
                        shot_maps_data_goals.groupby("is_home")["is_goal"].sum().reset_index(name="total_goals_by_location"),
                        on="is_home",
                        how="left"
                    )
                    .assign(team_share=lambda df: df["goal_count"] / df["total_goals_by_location"] * 100)
                    .pivot(index="team_name", columns="is_home", values="team_share")
                    .fillna(0)
                    .reset_index()
                    .melt(id_vars=["team_name"], var_name="is_home", value_name="team_share")
                )

        last_round = match_data_df['week'].max()

        create_goal_share_plot(
            team_goal_types_df,
            category,
            category_en,
            subcategory,
            last_round
        )

    except Exception as e:
        st.error("No suitable data found.")
        st.markdown(
            """
            <a href="https://github.com/urazakgul/datafc-web/issues" target="_blank" class="error-button">
                🛠️ Report Issue
            </a>
            """,
            unsafe_allow_html=True
        )