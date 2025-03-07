import streamlit as st
from modules.homepage import get_data
from code.funcs.team import (
    xg_time_series,
    xg_actual_vs_expected,
    xg_strengths_vs_weaknesses,
    xg_defensive_efficiency,
    team_performance,
    team_rating,
    team_similarity,
    goal_creation_patterns,
    team_win_rate,
    geometry,
    team_momentum_evolution
)
from code.utils.helpers import render_spinner
from config import match_performances, game_stats_group_name

def process_xg_analysis(team_list, situation_list, body_part_list):
    analysis_type = st.sidebar.selectbox(
        label="xG Analysis Type",
        options=[
            "Goals Over/Underperformance Compared to xG (Weekly Series)",
            "Actual vs Expected Goal Differences (Scored & Conceded)",
            "xG vs xGA",
            "Actual vs xG Scored & Conceded",
            "xG-Based Defensive Efficiency",
        ],
        index=None,
        label_visibility="hidden",
        placeholder="xG Analysis Type"
    )

    if analysis_type is None:
        st.warning("Please select an xG analysis.")
        return

    if analysis_type == "Goals Over/Underperformance Compared to xG (Weekly Series)":
        render_spinner(xg_time_series.main, team_list)
    elif analysis_type == "Actual vs Expected Goal Differences (Scored & Conceded)":
        render_spinner(xg_actual_vs_expected.main)
    elif analysis_type in [
        "xG vs xGA",
        "Actual vs xG Scored & Conceded",
    ]:

        selected_situation = None
        selected_body_part = None

        selected_category = st.sidebar.selectbox(
            label="Category",
            options=["All", "Situation", "Body Part"],
            index=None,
            label_visibility="hidden",
            placeholder="Category"
        )

        if selected_category is None:
            st.warning("Please select a category.")
            return

        if selected_category == "Situation":
            selected_situation = st.sidebar.selectbox(
                label="Situation",
                options=situation_list,
                index=None,
                label_visibility="hidden",
                placeholder="Situation"
            )
            if selected_situation is None:
                st.warning("Please select a situation.")
                return
            else:
                render_spinner(
                    xg_strengths_vs_weaknesses.main,
                    category=selected_category,
                    selected_situation=selected_situation,
                    selected_body_part=None,
                    plot_type=analysis_type
                )
        elif selected_category == "Body Part":
            selected_body_part = st.sidebar.selectbox(
                label="Body Part",
                options=body_part_list,
                index=None,
                label_visibility="hidden",
                placeholder="Body Part"
            )
            if selected_body_part is None:
                st.warning("Please select a body part.")
                return
            else:
                render_spinner(
                    xg_strengths_vs_weaknesses.main,
                    category=selected_category,
                    selected_situation=None,
                    selected_body_part=selected_body_part,
                    plot_type=analysis_type
                )
        elif selected_category == "All":
            render_spinner(
                xg_strengths_vs_weaknesses.main,
                category=selected_category,
                selected_situation=None,
                selected_body_part=None,
                plot_type=analysis_type
            )
    elif analysis_type == "xG-Based Defensive Efficiency":
        render_spinner(xg_defensive_efficiency.main)

def display_team_comparison():

    teams = get_data("standings_data")
    team_list = teams.loc[teams["category"] == "Total", "team_name"].sort_values().tolist()

    situation_and_body_part = get_data("shots_data")
    situation_list = sorted(s.capitalize() for s in situation_and_body_part["situation"].dropna().unique().tolist())
    body_part_list = sorted(s.capitalize() for s in situation_and_body_part["body_part"].dropna().unique().tolist())

    section = st.sidebar.selectbox(
        label="Category",
        options=[
            "xG (Expected Goals)",
            "Match Performance",
            "Rating",
            "Similarity",
            "Goal Creation Patterns",
            "Win Rate",
            "Geometry",
            "Momentum"
        ],
        index=None,
        label_visibility="hidden",
        placeholder="Category"
    )

    if section is None:
        st.warning("Please select a category.")
        return

    view_type = st.sidebar.radio(
        "View Type",
        ["Overall", "Weekly"],
        index=0,
        label_visibility="hidden"
    )

    if section == "xG (Expected Goals)":
        process_xg_analysis(team_list, situation_list, body_part_list)
    elif section == "Match Performance":
        subcategory = st.sidebar.selectbox(
            label="Statistic",
            options=match_performances,
            index=None,
            label_visibility="hidden",
            placeholder="Statistic"
        )
        if not subcategory:
            st.warning("Please select a statistic.")
            return
        render_spinner(team_performance.main, subcategory)
    elif section == "Rating":
        subcategory = st.sidebar.selectbox(
            label="Analysis Type",
            options=[
                "Mean-Standard Deviation (Overall)",
                "Mean-Standard Deviation (Home)",
                "Mean-Standard Deviation (Away)",
            ],
            index=None,
            label_visibility="hidden",
            placeholder="Analysis Type"
        )
        if not subcategory:
            st.warning("Please select an analysis type.")
            return
        render_spinner(team_rating.main, subcategory)
    elif section == "Similarity":
        similarity_algorithm = st.sidebar.selectbox(
            label="Similarity Algorithm",
            options=["Cosine Similarity", "Principal Component Analysis"],
            index=None,
            label_visibility="hidden",
            placeholder="Similarity Algorithm"
        )

        if similarity_algorithm is None:
            st.warning("Please select a similarity algorithm.")
            return

        if similarity_algorithm == "Cosine Similarity":
            selected_team = st.sidebar.selectbox(
                label="Team",
                options=team_list,
                index=None,
                label_visibility="hidden",
                placeholder="Team"
            )
        else:
            selected_team = None

        if similarity_algorithm == "Cosine Similarity" and not selected_team:
            st.warning("Please select a team.")
            return
        else:
            filtered_game_stats_group_name = [
                category for category in game_stats_group_name if category != "Match overview"
            ]
            selected_categories = st.sidebar.multiselect(
                label="Statistic Category",
                options=filtered_game_stats_group_name,
                default=filtered_game_stats_group_name
            )

            if not selected_categories:
                st.warning("Please select at least one statistic category.")
                return
            else:
                render_spinner(
                    team_similarity.main,
                    selected_team,
                    selected_categories,
                    similarity_algorithm
                )
    elif section == "Goal Creation Patterns":
        category = st.sidebar.selectbox(
            label="Goal Creation Pattern",
            options=[
                "Situation",
                "Body Part",
                "Time Interval",
                "Goal Mouth Location",
                "Player Position",
                "Home-Away"
            ],
            index=None,
            label_visibility="hidden",
            placeholder="Goal Creation Pattern"
        )
        if not category:
            st.warning("Please select a goal creation pattern.")
            return
        subcategory = st.sidebar.selectbox(
            label="Goal Share Type",
            options=["By Team Share", "By Team Comparison Share"],
            index=None,
            label_visibility="hidden",
            placeholder="Goal Share Type"
        )
        if not subcategory:
            st.warning("Please select a goal share type.")
            return
        render_spinner(
            goal_creation_patterns.main,
            category,
            subcategory
        )
    elif section == "Win Rate":
        render_spinner(
            team_win_rate.main
        )
    elif section == "Geometry":
        category = st.sidebar.selectbox(
            label="Geometric Analysis",
            options=[
                "Compactness",
                "Vertical Spread",
                "Horizontal Spread"
            ],
            index=None,
            label_visibility="hidden",
            placeholder="Geometric Analysis"
        )
        if not category:
            st.warning("Please select an analysis type.")
            return
        render_spinner(
            geometry.main,
            category
        )
    elif section == "Momentum":
        render_spinner(
            team_momentum_evolution.main,
            view_type
        )