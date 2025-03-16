import streamlit as st
import pandas as pd
from modules.homepage import get_data
from code.funcs.player import (
    player_heatmap,
    player_shot_location,
    player_rating
)
from code.utils.helpers import render_spinner, load_with_spinner, sort_turkish

def load_team_data(team, data_type):
    match_data_df = get_data("match_data")
    specific_data_df = get_data(data_type)

    match_data_df = match_data_df[["game_id", "tournament", "season", "week", "home_team", "away_team"]]

    if data_type == "coordinates_data":
        merged_data = specific_data_df.merge(
            match_data_df,
            on=["game_id", "tournament", "season", "week"],
            how="left"
        )
        merged_data["team_name"] = merged_data.apply(
            lambda row: row["home_team"] if row["team"] == "home" else row["away_team"],
            axis=1
        )
    elif data_type == "shots_data":
        specific_data_df = specific_data_df[[
            "tournament", "season", "week", "game_id", "player_name", "is_home", "shot_type", "goal_type",
            "situation", "goal_mouth_location", "player_coordinates_x", "player_coordinates_y"
        ]]
        merged_data = specific_data_df.merge(
            match_data_df,
            on=["tournament", "season", "week", "game_id"],
            how="left"
        )
        merged_data["team_name"] = merged_data.apply(
            lambda row: row["home_team"] if row["is_home"] else row["away_team"],
            axis=1
        )
        merged_data = merged_data[merged_data["goal_type"] != "own"]
    elif data_type == "lineups_data":
        merged_data = specific_data_df.merge(
            match_data_df,
            on=["tournament", "season", "week", "game_id"],
            how="left"
        )
        merged_data["team_name"] = merged_data.apply(
            lambda row: row["home_team"] if row["team"] == "home" else row["away_team"],
            axis=1
        )
        merged_data = merged_data[merged_data["stat_name"] == "rating"]

    team_data = merged_data[merged_data["team_name"] == team]
    return team_data[["player_name"]].drop_duplicates()

def handle_player_section(section):

    if st.session_state["selected_league"] == "super_lig":
        team_list = sort_turkish(pd.DataFrame({"team_name": get_data("standings_data")["team_name"].unique()}), column="team_name")["team_name"].tolist()
    else:
        team_list = sorted(get_data("standings_data")["team_name"].unique())

    situations = get_data("shots_data")
    situation_list = sorted([s.capitalize() for s in situations["situation"].dropna().unique() if s.lower() != "penalty"])

    selected_team = st.sidebar.selectbox(
        label="Team",
        options=team_list,
        index=None,
        label_visibility="hidden",
        placeholder="Team"
    )

    if not selected_team:
        st.warning("Please select a team.")
        return

    data_type = (
        "coordinates_data" if section == "Heatmap" else
        "shots_data" if section == "Shot Location" else
        "lineups_data" if section == "Rating" else None
    )

    team_data = load_with_spinner(load_team_data, selected_team, data_type)

    if st.session_state["selected_league"] == "super_lig":
        players_list = sort_turkish(pd.DataFrame({"player_name": team_data["player_name"].unique()}), column="player_name")["player_name"].tolist()
    else:
        players_list = sorted(team_data["player_name"].unique())

    selected_player = st.sidebar.selectbox(
        label="Player",
        options=players_list,
        index=None,
        label_visibility="hidden",
        placeholder="Player"
    )

    if not selected_player:
        st.warning("Please select a player.")
    else:
        if section == "Heatmap":
            render_spinner(player_heatmap.main, selected_team, selected_player)
        elif section == "Shot Location":
            show_xg_based = st.sidebar.radio(
                label="Show xG-based",
                options=["Standard View", "xG-Adjusted View"],
                index=None,
                horizontal=True,
                label_visibility="hidden"
            )

            if show_xg_based is None:
                st.warning("Please select a view option.")
                return

            include_shot_type = st.sidebar.radio(
                label="Shot Type Breakdown",
                options=["Overall Shot Outcome", "Break by Shot Type"],
                index=None,
                horizontal=True,
                label_visibility="hidden"
            )

            if include_shot_type is None:
                st.warning("Please select a shot type breakdown option.")
                return

            selected_situation = st.sidebar.selectbox(
                label="Situation",
                options=["All"] + situation_list,
                index=None,
                label_visibility="hidden",
                placeholder="Situation"
            )

            if not selected_situation:
                st.warning("Please select a situation.")
                return

            render_spinner(
                player_shot_location.main,
                selected_team,
                selected_player,
                selected_situation,
                show_xg_based,
                include_shot_type
            )
        elif section == "Rating":
            render_spinner(player_rating.main, selected_team, selected_player)

def display_player_based():
    section = st.sidebar.selectbox(
        label="Category",
        options=["Heatmap", "Shot Location", "Rating"],
        index=None,
        label_visibility="hidden",
        placeholder="Category"
    )

    if not section:
        st.warning("Please select a category.")
        return

    handle_player_section(section)