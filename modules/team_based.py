import streamlit as st
from modules.homepage import get_data
from code.funcs.team import (
    goal_network,
    team_shot_location
)
from code.utils.helpers import render_spinner

def handle_goal_network():

    teams = get_data("standings_data")
    team_list = teams.loc[teams["category"] == "Total", "team_name"].sort_values().tolist()

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

    side = st.sidebar.radio(
        label="For/Against",
        options=["For", "Against"],
        index=None,
        label_visibility="hidden"
    )

    if side is None:
        st.warning("Please select a side.")
        return

    plot_type = st.sidebar.radio(
        label="Map Type",
        options=["Combined", "Separated"],
        index=None,
        label_visibility="hidden"
    )

    if plot_type is None:
        st.warning("Please select a map type.")
        return

    render_spinner(
        goal_network.main,
        selected_team,
        plot_type,
        side
    )

def handle_shot_location():

    teams = get_data("standings_data")
    team_list = teams.loc[teams["category"] == "Total", "team_name"].sort_values().tolist()

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
    else:
        render_spinner(
            team_shot_location.main,
            selected_team,
            selected_situation
        )

def display_team_based():
    section = st.sidebar.selectbox(
        label="Category",
        options=["Goal Network", "Shot Location"],
        index=None,
        label_visibility="hidden",
        placeholder="Category"
    )

    if section is None:
        st.warning("Please select a category.")
        return

    if section == "Goal Network":
        handle_goal_network()
    elif section == "Shot Location":
        handle_shot_location()