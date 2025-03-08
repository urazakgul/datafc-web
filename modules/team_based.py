import streamlit as st
import pandas as pd
from modules.homepage import get_data
from code.funcs.team import (
    goal_network,
    team_shot_location
)
from code.utils.helpers import render_spinner, sort_turkish

def handle_goal_network():

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

    if plot_type == "Combined":
        combined_option = st.sidebar.radio(
            label="View Type",
            options=["Network with Heatmap", "Juego de Posici\u00f3n"],
            index=None,
            label_visibility="hidden"
        )
        if combined_option is None:
            st.warning("Please select a view type.")
            return
    else:
        combined_option = "Network with Heatmap"

    render_spinner(
        goal_network.main,
        selected_team,
        plot_type,
        side,
        combined_option
    )

def handle_shot_location():

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
    else:
        render_spinner(
            team_shot_location.main,
            selected_team,
            selected_situation,
            show_xg_based,
            include_shot_type
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