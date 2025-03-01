import streamlit as st
from code.funcs.match.xg_racer import main as xg_racer_main
from code.utils.helpers import render_spinner
from modules.homepage import get_data

def display_xg_analysis():

    analysis_type = st.sidebar.selectbox(
        label="xG Analysis Type",
        options=["xG Racer"],
        index=None,
        label_visibility="hidden",
        placeholder="xG Analysis Type"
    )

    if not analysis_type:
        st.warning("Please select an analysis type.")
        return

    if analysis_type == "xG Racer":
        match_data_df = get_data("match_data")
        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]
        match_data_df = match_data_df[["week", "home_team", "away_team"]]

        rounds = match_data_df["week"].unique()
        selected_round = st.sidebar.selectbox(
            label="Matchweeks",
            options=sorted(rounds),
            index=None,
            label_visibility="hidden",
            placeholder="Matchweeks"
        )

        if not selected_round:
            st.warning("Please select a matchweek.")
            return

        filtered_games = match_data_df[match_data_df["week"] == selected_round]
        team_pairs = filtered_games.apply(lambda row: f"{row['home_team']} - {row['away_team']}", axis=1)

        selected_match = st.sidebar.selectbox(
            label="Match",
            options=list(team_pairs),
            index=None,
            label_visibility="hidden",
            placeholder="Match"
        )

        if not selected_match:
            st.warning("Please select a match.")
            return

        home_team, away_team = selected_match.split(" - ")
        xg_racer_main(
            selected_round,
            home_team,
            away_team
        )

def display_match_comparison():
    section = st.sidebar.selectbox(
        label="Category",
        options=["xG (Expected Goals)"],
        index=None,
        label_visibility="hidden",
        placeholder="Category"
    )

    if not section:
        st.warning("Please select a category.")
        return

    if section == "xG (Expected Goals)":
        render_spinner(display_xg_analysis)