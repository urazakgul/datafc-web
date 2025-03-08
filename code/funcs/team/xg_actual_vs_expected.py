import streamlit as st
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_actual_vs_expected_xg_plot(xg_xga_gerceklesen_teams):

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(
        xg_xga_gerceklesen_teams["xgDiff"],
        xg_xga_gerceklesen_teams["xgConcededDiff"],
        alpha=0
    )

    mean_xgDiff = 0
    mean_xgConcededDiff = 0

    ax.axhline(y=mean_xgConcededDiff, color="darkred", linestyle="--", linewidth=2, label="Actual - xGA = 0")
    ax.axvline(x=mean_xgDiff, color="darkblue", linestyle="--", linewidth=2, label="Actual - xG = 0")

    def getImage(path):
        return OffsetImage(plt.imread(path), zoom=.3, alpha=1)

    for index, row in xg_xga_gerceklesen_teams.iterrows():
        logo_path = f"./imgs/{st.session_state['selected_league']}_logos/{row['team_name']}.png"
        ab = AnnotationBbox(getImage(logo_path), (row["xgDiff"], row["xgConcededDiff"]), frameon=False)
        ax.add_artist(ab)

    ax.set_xlabel("Actual - xG (Higher is better)", labelpad=20, fontsize=12)
    ax.set_ylabel("Actual - xGA (Lower is better)", labelpad=20, fontsize=12)
    ax.set_title(
        f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Actual vs Expected Goal Differences (Scored & Conceded) by Team",
        fontsize=14,
        fontweight="bold",
        pad=40
    )
    ax.grid(True, linestyle="--", alpha=0.7)
    add_footer(fig)
    ax.invert_yaxis()

    st.pyplot(fig)

def main():
    try:

        match_data_df = get_data("match_data")
        shots_data_df = get_data("shots_data")
        standings_data_df = get_data("standings_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]

        standings_data_df = standings_data_df[standings_data_df["category"] == "Total"][["team_name", "scores_for", "scores_against"]]

        shots_data_df = shots_data_df.merge(match_data_df, on=["tournament", "season", "week", "game_id"])
        shots_data_df["team_name"] = shots_data_df.apply(lambda x: x["home_team"] if x["is_home"] else x["away_team"], axis=1)

        xg_xga_df = shots_data_df.groupby(["game_id","team_name"])["xg"].sum().reset_index()

        for game_id in xg_xga_df["game_id"].unique():
            game_data = xg_xga_df[xg_xga_df["game_id"] == game_id]
            match_info = match_data_df[match_data_df["game_id"] == game_id]

            if not match_info.empty:
                home_team = match_info["home_team"].values[0]
                away_team = match_info["away_team"].values[0]

                for index, row in game_data.iterrows():
                    opponent_xg = game_data.loc[game_data["team_name"] != row["team_name"], "xg"].values

                    if opponent_xg.size > 0:
                        xg_xga_df.at[index, "xga"] = opponent_xg[0]
                    else:
                        if row["team_name"] not in [home_team, away_team]:
                            xg_xga_df.at[index, "xga"] = 0

        team_totals_df = xg_xga_df.groupby("team_name")[["xg", "xga"]].sum().reset_index()
        xg_xga_teams = team_totals_df.rename(columns={"xga":"xgConceded"})

        actual_xg_xga_diffs = standings_data_df.merge(
            xg_xga_teams,
            on="team_name"
        )
        actual_xg_xga_diffs["xgDiff"] = actual_xg_xga_diffs["scores_for"] - actual_xg_xga_diffs["xg"]
        actual_xg_xga_diffs["xgConcededDiff"] = actual_xg_xga_diffs["scores_against"] - actual_xg_xga_diffs["xgConceded"]
        xg_xga_gerceklesen_teams = actual_xg_xga_diffs[["team_name","xgDiff","xgConcededDiff"]]

        create_actual_vs_expected_xg_plot(xg_xga_gerceklesen_teams)

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