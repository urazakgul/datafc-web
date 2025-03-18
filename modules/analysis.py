import streamlit as st
from modules.homepage import get_data
from code.funcs.analytics import (
    match_statistics_impact_analysis,
    scoring_analysis,
    predictive_analytics
)
from code.utils.helpers import render_spinner
from config import match_performance_binary

def load_game_data(selected_week):
    match_data_df = get_data("match_data")
    games_data_selected_week = match_data_df[match_data_df["week"] == selected_week]

    if not games_data_selected_week.empty:
        return games_data_selected_week
    else:
        st.warning("No matches found for the selected matchweek in the league and season.")
        return None

def handle_eda_analysis(extended_options):
    selected_variable = st.sidebar.selectbox(
        label="Variable",
        options=extended_options,
        index=None,
        label_visibility="hidden",
        placeholder="Variable"
    )

    if selected_variable is None:
        st.warning("Please select a variable.")
        return

    render_spinner(
        match_statistics_impact_analysis.main,
        selected_variable
    )

def handle_scoring_analysis(selected_scoring_analysis):
    render_spinner(
        scoring_analysis.main,
        selected_scoring_analysis
    )

def handle_predictive_analytics(selected_model):
    match_data_df = get_data("match_data")
    filtered_weeks_list = sorted(set(match_data_df.loc[match_data_df["status"] == "Not started", "week"].tolist()))
    selected_week = min(filtered_weeks_list)

    games_data_selected_week = load_game_data(selected_week)
    if games_data_selected_week is None:
        return

    games_list = [f"{row['home_team']} - {row['away_team']}" for index, row in games_data_selected_week.iterrows()]
    selected_game = st.sidebar.selectbox(
        label="Upcoming Matches",
        options=games_list,
        index=None,
        label_visibility="hidden",
        placeholder="Upcoming Matches"
    )

    if selected_game is None:
        st.warning("Please select a match.")
        return

    if selected_model == "Dixon-Coles":
        plot_type = st.sidebar.radio(
            label="Visualization Type",
            options=["Matrix", "Ranked", "Summary", "Team Strength"],
            index=None,
            label_visibility="hidden"
        )

        if plot_type is None:
            st.warning("Please select a visualization type.")
            return

        if plot_type == "Summary":
            first_n_goals = st.sidebar.number_input(
                label="Goal Combinations",
                min_value=2,
                max_value=121,
                value=10,
                label_visibility="hidden",
                placeholder="Goal Combinations"
            )
        else:
            first_n_goals = 10
    elif selected_model == "Bradley-Terry":
        plot_type = None
        first_n_goals = None

    render_spinner(
        predictive_analytics.main,
        selected_model,
        selected_game,
        selected_week,
        plot_type,
        first_n_goals
    )

def display_eda_analysis():
    match_performances = get_data("match_stats_data")
    match_performance_list = sorted(match_performances["stat_name"].unique())

    extended_options = []
    for stat in match_performance_list:
        if stat in match_performance_binary:
            extended_options.append(f"{stat} (Success)")
            extended_options.append(f"{stat} (Total)")
        else:
            extended_options.append(stat)

    selected_category = st.sidebar.selectbox(
        label="Category",
        options=["Impact of Statistics on Matches"],
        index=None,
        label_visibility="hidden",
        placeholder="Category"
    )

    if selected_category is None:
        st.warning("Please select a category.")
        return

    handle_eda_analysis(selected_category, extended_options)

def display_scoring_analysis():
    selected_scoring_analysis = st.sidebar.selectbox(
        label="Scoring Analysis",
        options=["Home vs Away Goal Matrix"],
        index=None,
        label_visibility="hidden",
        placeholder="Scoring Analysis Type"
    )

    if selected_scoring_analysis is None:
        st.warning("Please select a scoring analysis type.")
        return

    handle_scoring_analysis(selected_scoring_analysis)

def display_predictive_analytics():
    selected_model = st.sidebar.selectbox(
        label="Prediction Method",
        options=["Dixon-Coles", "Bradley-Terry"],
        index=None,
        label_visibility="hidden",
        placeholder="Prediction Method"
    )

    if selected_model is None:
        st.warning("Please select a prediction method.")
        return

    handle_predictive_analytics(selected_model)