import numpy as np
import streamlit as st
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_strength_vs_weakness_xg_plot(xg_xga_sw_teams, plot_type, category=None, situation_type=None, body_part_type=None):

    fig, ax = plt.subplots(figsize=(12, 12))

    if plot_type == "xG vs xGA":
        xg_col, xga_col = "xg", "xga"
        title_suffix = "xG vs xGA by Team"
        label_suffix_1 = "xG (Higher is better)"
        label_suffix_2 = "xGA (Lower is better)"
    elif plot_type == "Actual vs xG Scored & Conceded":
        xg_col, xga_col = "xgDiff", "xgaDiff"
        title_suffix = "Actual vs xG Scored & Conceded by Team"
        label_suffix_1 = "Goals Scored - xG"
        label_suffix_2 = "Goals Conceded - xGA"

    if category == "All":
        info_text = ""
    elif situation_type is not None:
        info_text = f"\n(Situation | {situation_type})"
    elif body_part_type is not None:
        info_text = f"\n(Body Part | {body_part_type})"

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(
        xg_xga_sw_teams[xg_col],
        xg_xga_sw_teams[xga_col],
        alpha=0
    )

    mean_xgDiff = xg_xga_sw_teams[xg_col].mean()
    mean_xgConcededDiff = xg_xga_sw_teams[xga_col].mean()

    ax.axhline(y=mean_xgConcededDiff, color="darkred", linestyle="--", linewidth=2, label="Actual - xGA = 0")
    ax.axvline(x=mean_xgDiff, color="darkblue", linestyle="--", linewidth=2, label="Actual - xG = 0")

    def getImage(path):
        return OffsetImage(plt.imread(path), zoom=.3, alpha=1)

    for index, row in xg_xga_sw_teams.iterrows():
        logo_path = f"./imgs/{st.session_state['selected_league']}_logos/{row['team_name']}.png"
        ab = AnnotationBbox(getImage(logo_path), (row[xg_col], row[xga_col]), frameon=False)
        ax.add_artist(ab)

    ax.set_xlabel(label_suffix_1, labelpad=20, fontsize=12)
    ax.set_ylabel(label_suffix_2, labelpad=20, fontsize=12)
    ax.set_title(
        f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – {title_suffix}\n{info_text}",
        fontsize=14,
        fontweight="bold",
        pad=40
    )
    ax.grid(True, linestyle="--", alpha=0.7)
    add_footer(fig)
    ax.invert_yaxis()

    st.pyplot(fig)

def main(category=None, selected_situation=None, selected_body_part=None, plot_type=None):
    try:

        match_data_df = get_data("match_data")
        shots_data_df = get_data("shots_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]

        shots_data_df = shots_data_df.merge(match_data_df, on=["tournament", "season", "week", "game_id"])
        shots_data_df["team_name"] = shots_data_df.apply(lambda x: x["home_team"] if x["is_home"] else x["away_team"], axis=1)

        if category == "Situation" and selected_situation is not None:
            filtered_data = shots_data_df[shots_data_df["situation"] == selected_situation.lower()]
            grouping_columns = ["game_id", "team_name", "situation"]
        elif category == "Body Part" and selected_body_part is not None:
            filtered_data = shots_data_df[shots_data_df["body_part"] == selected_body_part.lower()]
            grouping_columns = ["game_id", "team_name", "body_part"]
        elif category == "All":
            filtered_data = shots_data_df
            grouping_columns = ["game_id", "team_name"]

        xg_xga_df = filtered_data.groupby(grouping_columns)["xg"].sum().reset_index()

        goals_data = filtered_data[filtered_data["shot_type"] == "goal"]
        goals_df = goals_data.groupby(grouping_columns)["shot_type"].count().reset_index()
        goals_df = goals_df.rename(columns={"shot_type": "goals"})

        xg_xga_df = xg_xga_df.merge(goals_df, on=grouping_columns, how="left")
        xg_xga_df["goals"] = xg_xga_df["goals"].fillna(0)

        for game_id in xg_xga_df["game_id"].unique():
            game_data = xg_xga_df[xg_xga_df["game_id"] == game_id]
            match_info = match_data_df[match_data_df["game_id"] == game_id]

            if not match_info.empty:
                home_team = match_info["home_team"].values[0]
                away_team = match_info["away_team"].values[0]

                for index, row in game_data.iterrows():
                    opponent_xg = game_data.loc[game_data["team_name"] != row["team_name"], "xg"].values
                    opponent_goals = game_data.loc[game_data["team_name"] != row["team_name"], "goals"].values

                    if opponent_xg.size > 0:
                        xg_xga_df.at[index, "xga"] = opponent_xg[0]
                    else:
                        if row["team_name"] not in [home_team, away_team]:
                            xg_xga_df.at[index, "xga"] = 0

                    if opponent_goals.size > 0:
                        xg_xga_df.at[index, "conceded_goals"] = opponent_goals[0]
                    else:
                        if row["team_name"] not in [home_team, away_team]:
                            xg_xga_df.at[index, "conceded_goals"] = 0

        grouping_columns_without_game_id = [col for col in grouping_columns if col != "game_id"]
        team_totals_df = xg_xga_df.groupby(grouping_columns_without_game_id)[["xg", "xga", "goals", "conceded_goals"]].sum().reset_index()

        team_totals_df["xgDiff"] = team_totals_df["goals"] - team_totals_df["xg"]
        team_totals_df["xgaDiff"] = team_totals_df["conceded_goals"] - team_totals_df["xga"]

        if plot_type == "xG vs xGA":
            create_strength_vs_weakness_xg_plot(
                team_totals_df,
                plot_type="xG vs xGA",
                category=category,
                situation_type=selected_situation,
                body_part_type=selected_body_part
            )
        elif plot_type == "Actual vs xG Scored & Conceded":
            create_strength_vs_weakness_xg_plot(
                team_totals_df,
                plot_type="Actual vs xG Scored & Conceded",
                category=category,
                situation_type=selected_situation,
                body_part_type=selected_body_part
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