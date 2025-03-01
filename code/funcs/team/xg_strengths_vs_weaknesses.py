import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_strength_vs_weakness_xg_plot(xg_xga_sw_teams, last_round, plot_type, category=None, situation_type=None, body_part_type=None):

    fig, ax = plt.subplots(figsize=(12, 12))

    if plot_type == "xG vs xGA":
        x_col, xga_col = "xg", "xga"
        title_suffix = "xG vs xGA per Team"
        label_suffix_1 = "xG"
        label_suffix_2 = "xGA"
    elif plot_type == "Actual vs xG Scored & Conceded":
        x_col, xga_col = "xgDiff", "xgaDiff"
        title_suffix = "Actual vs xG Scored & Conceded per Team"
        label_suffix_1 = "Goals Scored - xG"
        label_suffix_2 = "Goals Conceded - xGA"

    if category == "All":
        info_text = ""
    elif situation_type is not None:
        info_text = f"\n(Situation | {situation_type})"
    elif body_part_type is not None:
        info_text = f"\n(Body Part | {body_part_type})"

    xg_xga_sw_teams = xg_xga_sw_teams.sort_values(x_col, ascending=True)

    teams = xg_xga_sw_teams["team_name"]
    xg_values = xg_xga_sw_teams[x_col]
    xga_values = xg_xga_sw_teams[xga_col]

    y = np.arange(len(teams))

    for i, (xg_val, xga_val) in enumerate(zip(xg_values, xga_values)):
        ax.plot([xg_val, xga_val], [y[i], y[i]], color="gray", alpha=0.5, linewidth=1)

    for i, val in enumerate(xg_values):
        ax.scatter(val, y[i], color="blue", edgecolors='black', s=150, label=label_suffix_1 if i == 0 else "")

    for i, val in enumerate(xga_values):
        ax.scatter(val, y[i], color="red", edgecolors='black', s=150, label=label_suffix_2 if i == 0 else "")

    ax.set_yticks(y)
    ax.set_yticklabels(teams, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

    ax.set_title(
        f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – {title_suffix}\n{info_text}",
        fontsize=14,
        fontweight="bold",
        pad=35
    )
    add_footer(fig)
    ax.set_xlabel("")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False, fontsize=10)

    plt.tight_layout()

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

        last_round = match_data_df["week"].max()

        if plot_type == "xG vs xGA":
            create_strength_vs_weakness_xg_plot(
                team_totals_df,
                last_round,
                plot_type="xG vs xGA",
                category=category,
                situation_type=selected_situation,
                body_part_type=selected_body_part
            )
        elif plot_type == "Actual vs xG Scored & Conceded":
            create_strength_vs_weakness_xg_plot(
                team_totals_df,
                last_round,
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