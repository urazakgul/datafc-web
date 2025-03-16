import streamlit as st
import pandas as pd
from modules.homepage import get_data
from code.funcs.player import (
    player_goal_sequence_involvement
)
from code.utils.helpers import render_spinner, sort_turkish

def handle_player_section(section):

    if st.session_state["selected_league"] == "super_lig":
        team_list = sort_turkish(pd.DataFrame({"team_name": get_data("standings_data")["team_name"].unique()}), column="team_name")["team_name"].tolist()
    else:
        team_list = sorted(get_data("standings_data")["team_name"].unique())

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

    if section == "Goal Sequence Involvement":
        render_spinner(player_goal_sequence_involvement.main, selected_team)
        return

def display_player_comparison():
    section = st.sidebar.selectbox(
        label="Category",
        options=["Goal Sequence Involvement"],
        index=None,
        label_visibility="hidden",
        placeholder="Category"
    )

    if not section:
        st.warning("Please select a category.")
        return

    handle_player_section(section)