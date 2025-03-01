import pandas as pd
import streamlit as st
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer, sort_turkish
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_xg_cum_actual_plot(xg_goal_teams, teams, plot_type):

    global_x_min = xg_goal_teams["week"].min()
    global_x_max = xg_goal_teams["week"].max()

    if plot_type == "Cumulative xG vs Goals Scored (Weekly Series)":
        global_y_min = min(xg_goal_teams["cumulative_goal_count"].min(), xg_goal_teams["cumulative_total_xg"].min())
        global_y_max = max(xg_goal_teams["cumulative_goal_count"].max(), xg_goal_teams["cumulative_total_xg"].max())
    elif plot_type == "Goals Over/Underperformance Compared to xG (Weekly Series)":
        global_y_min = xg_goal_teams["cum_goal_xg_diff"].min()
        global_y_max = xg_goal_teams["cum_goal_xg_diff"].max()

    fig, axes = plt.subplots(
        nrows=(len(teams) + 3) // 4,
        ncols=4,
        figsize=(20, 5 * ((len(teams) + 3) // 4))
    )
    axes = axes.flatten()

    for i, team in enumerate(teams):
        team_data = xg_goal_teams[xg_goal_teams["team_name"] == team].copy()

        ax = axes[i]

        if plot_type == "Cumulative xG vs Goals Scored (Weekly Series)":
            ax.plot(team_data["week"], team_data["cumulative_goal_count"], label="Cumulative Goals Scored", color="blue")
            ax.plot(team_data["week"], team_data["cumulative_total_xg"], label="Cumulative xG", color="red")
            ax.fill_between(
                team_data["week"],
                team_data["cumulative_goal_count"],
                team_data["cumulative_total_xg"],
                where=(team_data["cumulative_goal_count"] >= team_data["cumulative_total_xg"]),
                color="blue",
                alpha=0.3,
                interpolate=True
            )
            ax.fill_between(
                team_data["week"],
                team_data["cumulative_goal_count"],
                team_data["cumulative_total_xg"],
                where=(team_data["cumulative_goal_count"] < team_data["cumulative_total_xg"]),
                color="red",
                alpha=0.3,
                interpolate=True
            )
            fig.legend(
                ["Cumulative Goals Scored", "Cumulative xG"],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.00),
                frameon=False,
                ncol=2,
                fontsize="large"
            )
            fig.suptitle(
                f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Cumulative xG vs Goals Scored for Each Team",
                fontsize=24,
                fontweight="bold",
                y=1.02
            )
            ax.grid(True, linestyle="--", alpha=0.7)
        elif plot_type == "Goals Over/Underperformance Compared to xG (Weekly Series)":
            diff = team_data["cumulative_goal_count"] - team_data["cumulative_total_xg"]
            team_data["diff"] = diff.round(5)
            for i in range(len(team_data) - 1):
                x = team_data["week"].iloc[i:i+2]
                y = team_data["diff"].iloc[i:i+2]

                if y.mean() >= 0:
                    ax.plot(x, y, color="darkblue", linewidth=3)
                else:
                    ax.plot(x, y, color="darkred", linewidth=3)

            ax.fill_between(
                team_data["week"],
                0,
                team_data["diff"],
                where=(team_data["diff"] >= 0),
                color="blue",
                alpha=0.3,
                interpolate=True
            )
            ax.fill_between(
                team_data["week"],
                0,
                team_data["diff"],
                where=(team_data["diff"] < 0),
                color="red",
                alpha=0.3,
                interpolate=True
            )

            positive_patch = mpatches.Patch(color="darkblue", label="Goals Overperformed (Scored More Than xG)")
            negative_patch = mpatches.Patch(color="darkred", label="Goals Underperformed (Scored Less Than xG)")

            fig.legend(
                handles=[positive_patch, negative_patch],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.00),
                frameon=False,
                ncol=2,
                fontsize="large"
            )

            fig.suptitle(
                f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Goals Over/Underperformance Compared to xG for Each Team",
                fontsize=24,
                fontweight="bold",
                y=1.02
            )
            ax.grid(True, linestyle="--", alpha=0.7)

        ax.set_title(team)

        ax.xaxis.set_major_locator(MultipleLocator(3))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)
        ax.set_title(team)

    for j in range(len(teams), len(axes)):
        fig.delaxes(axes[j])

    fig.text(0.5, 0.04, "Week", ha="center", va="center", fontsize="large")
    add_footer(fig, y=0.02, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    st.pyplot(fig)

def main(team_list, plot_type):
    try:

        match_data_df = get_data("match_data")
        shots_data_df = get_data("shots_data")
        standings_data_df = get_data("standings_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]

        standings_data_df = standings_data_df[standings_data_df["category"] == "Total"][["team_name", "scores_for", "scores_against"]]

        shots_data_df = shots_data_df.merge(match_data_df, on=["tournament", "season", "week", "game_id"])
        shots_data_df["team_name"] = shots_data_df.apply(lambda x: x["home_team"] if x["is_home"] else x["away_team"], axis=1)

        xg_by_team = shots_data_df.groupby(["team_name", "week"])["xg"].sum().reset_index(name="total_xg")
        xg_by_team_pivot = xg_by_team.pivot(index="team_name", columns="week", values="total_xg").fillna(0)
        xg_by_team_long = xg_by_team_pivot.reset_index().melt(id_vars="team_name", var_name="week", value_name="total_xg")

        goal_shots_by_team_long = (
            match_data_df
            .melt(id_vars=["game_id", "week"], value_vars=["home_team", "away_team"], var_name="team_type", value_name="team")
            .merge(
                match_data_df.melt(id_vars=["game_id", "week"], value_vars=["home_score_display", "away_score_display"], var_name="goal_type", value_name="goals"),
                on=["game_id", "week"]
            )
            .query("team_type.str.replace('_team', '') == goal_type.str.replace('_score_display', '')")
            .groupby(["team", "week"], as_index=False)["goals"].sum()
            .rename(columns={
                "team": "team_name",
                "goals": "week_goal_count"
            })
        )
        goal_shots_by_team_long["week_goal_count"] = pd.to_numeric(goal_shots_by_team_long["week_goal_count"], errors="coerce")

        xg_goal_teams = pd.merge(xg_by_team_long, goal_shots_by_team_long, on=["team_name", "week"])
        xg_goal_teams = xg_goal_teams.sort_values(by=["team_name", "week"])
        xg_goal_teams["cumulative_total_xg"] = xg_goal_teams.groupby("team_name")["total_xg"].cumsum()
        xg_goal_teams["cumulative_goal_count"] = xg_goal_teams.groupby("team_name")["week_goal_count"].cumsum()
        xg_goal_teams["cum_goal_xg_diff"] = xg_goal_teams["cumulative_goal_count"] - xg_goal_teams["cumulative_total_xg"]

        xg_goal_teams["cumulative_goal_count"] = pd.to_numeric(xg_goal_teams["cumulative_goal_count"], errors="coerce")
        xg_goal_teams["cumulative_total_xg"] = pd.to_numeric(xg_goal_teams["cumulative_total_xg"], errors="coerce")
        xg_goal_teams["week"] = pd.to_numeric(xg_goal_teams["week"], errors="coerce")

        xg_goal_teams = xg_goal_teams.dropna(subset=["cumulative_goal_count", "cumulative_total_xg", "week"])

        teams_df = pd.DataFrame({"team_name": team_list})
        if st.session_state["selected_league"] == "super_lig":
            teams_sorted = sort_turkish(teams_df, "team_name")["team_name"].tolist()
        else:
            teams_sorted = teams_df.sort_values("team_name")["team_name"].tolist()

        teams = [team for team in teams_sorted if team in xg_goal_teams["team_name"].unique()]

        if plot_type == "Cumulative xG vs Goals Scored (Weekly Series)":
            create_xg_cum_actual_plot(xg_goal_teams, teams, plot_type="Cumulative xG vs Goals Scored (Weekly Series)")
        elif plot_type == "Goals Over/Underperformance Compared to xG (Weekly Series)":
            create_xg_cum_actual_plot(xg_goal_teams, teams, plot_type="Goals Over/Underperformance Compared to xG (Weekly Series)")

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