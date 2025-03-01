import streamlit as st
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_xg_defence_efficiency_plot(team_opponent_df, last_round):

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(
        team_opponent_df["non_penalty_xg_per_shot_against"],
        team_opponent_df["non_penalty_shot_conversion_against"],
        alpha=0
    )

    mean_non_penalty_xg_per_shot_against = team_opponent_df["non_penalty_xg_per_shot_against"].mean()
    mean_non_penalty_shot_conversion_against = team_opponent_df["non_penalty_shot_conversion_against"].mean()

    ax.axvline(x=mean_non_penalty_xg_per_shot_against, color="darkblue", linestyle="--", linewidth=2, label="Opponent xG per Shot (Average)")
    ax.axhline(y=mean_non_penalty_shot_conversion_against, color="darkred", linestyle="--", linewidth=2, label="Opponent Shot Conversion Rate (%) (Average)")

    def getImage(path):
        return OffsetImage(plt.imread(path), zoom=.3, alpha=1)

    for index, row in team_opponent_df.iterrows():
        logo_path = f"./imgs/{st.session_state['selected_league']}_logos/{row['team_name']}.png"
        ab = AnnotationBbox(getImage(logo_path), (row["non_penalty_xg_per_shot_against"], row["non_penalty_shot_conversion_against"]), frameon=False)
        ax.add_artist(ab)

    ax.set_xlabel("Opponent xG per Shot (Non-Penalty) (Lower is better)", labelpad=20, fontsize=12)
    ax.set_ylabel("Opponent Shot Conversion Rate (%) (Non-Penalty) (Lower is better)", labelpad=20, fontsize=12)
    ax.set_title(
        f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Opponent Shot Quality and Shot Conversion Rate",
        fontsize=14,
        fontweight="bold",
        pad=40
    )
    ax.grid(True, linestyle="--", alpha=0.7)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        fontsize=10,
        frameon=False
    )

    ax.invert_xaxis()
    ax.invert_yaxis()

    add_footer(fig)

    st.pyplot(fig)

def main():
    try:

        match_data_df = get_data("match_data")
        shots_data_df = get_data("shots_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]

        shots_data_df = shots_data_df.merge(match_data_df, on=["tournament", "season", "week", "game_id"])
        shots_data_df["team_name"] = shots_data_df.apply(lambda x: x["home_team"] if x["is_home"] else x["away_team"], axis=1)
        shots_data_df = shots_data_df[shots_data_df["goal_type"] != "penalty"]
        shots_data_df["is_goal"] = shots_data_df["shot_type"].apply(lambda x: 1 if x == "goal" else 0)

        xg_xga_df = shots_data_df.groupby(["game_id", "team_name"]).agg(
            xg=("xg", "sum"),
            shots=("xg", "count"),
            goals=("is_goal", "sum")
        ).reset_index()

        for game_id in xg_xga_df["game_id"].unique():
            game_data = xg_xga_df[xg_xga_df["game_id"] == game_id]
            match_info = match_data_df[match_data_df["game_id"] == game_id]

            if not match_info.empty:
                home_team = match_info["home_team"].values[0]
                away_team = match_info["away_team"].values[0]

                for index, row in game_data.iterrows():
                    opponent_data = game_data[game_data["team_name"] != row["team_name"]]

                    if not opponent_data.empty:
                        opponent_xg = opponent_data["xg"].values[0]
                        opponent_shots = opponent_data["shots"].values[0]
                        opponent_goals = opponent_data["goals"].values[0]

                        xg_xga_df.at[index, "xga"] = opponent_xg
                        xg_xga_df.at[index, "opponent_shots"] = opponent_shots
                        xg_xga_df.at[index, "opponent_goals"] = opponent_goals
                    else:
                        if row["team_name"] not in [home_team, away_team]:
                            xg_xga_df.at[index, "xga"] = 0
                            xg_xga_df.at[index, "opponent_shots"] = 0
                            xg_xga_df.at[index, "opponent_goals"] = 0

        team_opponent_df = xg_xga_df.groupby("team_name").agg(
            xg=("xg", "sum"),
            xgConceded=("xga", "sum"),
            shots=("shots", "sum"),
            goals=("goals", "sum"),
            opponent_shots=("opponent_shots", "sum"),
            opponent_goals=("opponent_goals", "sum")
        ).reset_index()

        team_opponent_df['non_penalty_xg_per_shot_against'] = team_opponent_df['xgConceded'] / team_opponent_df['opponent_shots']
        team_opponent_df['non_penalty_shot_conversion_against'] = (team_opponent_df['opponent_goals'] / team_opponent_df['opponent_shots']) * 100

        last_round = match_data_df["week"].max()

        create_xg_defence_efficiency_plot(team_opponent_df, last_round)

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