import streamlit as st
import pandas as pd
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer, sort_turkish
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_momentum_evolution_plot(final_summary_df, view_type):

    if view_type == "Overall":
        fig, ax = plt.subplots(figsize=(12, 10))

        ax.scatter(
            final_summary_df["team_momentum_per"],
            final_summary_df["opponent_momentum_per"],
            alpha=0
        )

        mean_team_momentum_per = final_summary_df["team_momentum_per"].mean()
        mean_opponent_momentum_per = final_summary_df["opponent_momentum_per"].mean()

        ax.axvline(x=mean_team_momentum_per, color="darkblue", linestyle="--", linewidth=2, label="League Average (Teams)")
        ax.axhline(y=mean_opponent_momentum_per, color="darkred", linestyle="--", linewidth=2, label="League Average (Opponents)")

        def getImage(path):
            return OffsetImage(plt.imread(path), zoom=.3, alpha=1)

        for index, row in final_summary_df.iterrows():
            logo_path = f"./imgs/{st.session_state['selected_league']}_logos/{row['team']}.png"
            ab = AnnotationBbox(getImage(logo_path), (row["team_momentum_per"], row["opponent_momentum_per"]), frameon=False)
            ax.add_artist(ab)

        ax.set_xlabel("Teams' Momentum Productivity (Higher is better)", labelpad=20, fontsize=12)
        ax.set_ylabel("Opponents' Momentum Productivity (Lower is better)", labelpad=20, fontsize=12)
        ax.set_title(
            f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Teams' vs. Opponents' Momentum Productivity",
            fontsize=14,
            fontweight="bold",
            pad=40
        )
        ax.grid(True, linestyle="--", alpha=0.7)
        add_footer(fig)
        ax.invert_yaxis()

    elif view_type == "Weekly":
        if st.session_state["selected_league"] == "super_lig":
            sorted_teams = sort_turkish(pd.DataFrame({"team": final_summary_df["team"].unique()}), column="team")["team"].tolist()
        else:
            sorted_teams = sorted(final_summary_df["team"].unique())

        fig, axes = plt.subplots(
            nrows=(len(sorted_teams) + 3) // 4,
            ncols=4,
            figsize=(20, 5 * ((len(sorted_teams) + 3) // 4))
        )
        axes = axes.flatten()

        for i, team in enumerate(sorted_teams):
            team_data = final_summary_df[final_summary_df["team"] == team].copy()
            ax = axes[i]

            ax.plot(team_data["week"], team_data["team_momentum_per"], label="Team Momentum Productivity", color="blue", marker="o")
            ax.plot(team_data["week"], team_data["opponent_momentum_per"], label="Opponent Momentum Productivity", color="darkred", marker="o")

            ax.set_title(team, pad=35)
            ax.grid(True)

            avg_team_rate = team_data["team_momentum_per"].mean() if not team_data.empty else 0
            avg_opponent_rate = team_data["opponent_momentum_per"].mean() if not team_data.empty else 0
            ax.text(
                x=0.5,
                y=1.05,
                s=f"Team: {avg_team_rate:.1f}, Opponent: {avg_opponent_rate:.1f}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14
            )

            ax.xaxis.set_major_locator(MultipleLocator(3))
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

            ax.grid(True, linestyle="--", alpha=0.7)

        for j in range(len(sorted_teams), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(
            f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Weekly Momentum Productivity by Team",
            fontsize=24,
            fontweight="bold",
            y=1.00
        )

        fig.text(0.5, 0.04, "Week", ha="center", va="center", fontsize=16)
        fig.text(-0.04, 0.5, "Momentum Productivity", ha="center", va="center", rotation="vertical", fontsize=16)

        fig.legend(
            ["Team Momentum", "Opponent Momentum"],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            frameon=False,
            ncol=2,
            fontsize=16
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        add_footer(fig, y=0.01, fontsize=14)

    st.pyplot(fig)

def main(view_type):
    try:

        match_data_df = get_data("match_data")
        momentum_data_df = get_data("momentum_data")

        match_data_df = match_data_df[["country","tournament","season","week","game_id","home_team","away_team"]]

        match_momentum_data_df = momentum_data_df.merge(
            match_data_df,
            on=["country","tournament","season","week","game_id"],
            how="left"
        )

        match_momentum_data_df.sort_values(by=["game_id", "minute"], inplace=True)

        match_momentum_data_df["home_team_momentum_count"] = match_momentum_data_df.groupby(
            ["country", "tournament", "season", "week", "game_id", "home_team"]
        )["value"].transform(lambda x: (x > 0).astype(int).cumsum())

        match_momentum_data_df["away_team_momentum_count"] = match_momentum_data_df.groupby(
            ["country", "tournament", "season", "week", "game_id", "away_team"]
        )["value"].transform(lambda x: (x < 0).astype(int).cumsum())

        match_momentum_data_df["home_team_momentum_value"] = match_momentum_data_df.groupby(
            ["country", "tournament", "season", "week", "game_id", "home_team"]
        )["value"].transform(lambda x: x.where(x > 0, 0).abs().cumsum())

        match_momentum_data_df["away_team_momentum_value"] = match_momentum_data_df.groupby(
            ["country", "tournament", "season", "week", "game_id", "away_team"]
        )["value"].transform(lambda x: x.where(x < 0, 0).abs().cumsum())

        home_last_momentum = match_momentum_data_df.groupby(
            ["country", "tournament", "season", "week", "home_team"]
        ).last().reset_index()[[
            "country", "tournament", "season", "week",
            "home_team", "home_team_momentum_count", "home_team_momentum_value",
            "away_team", "away_team_momentum_count", "away_team_momentum_value"
        ]]

        away_last_momentum = match_momentum_data_df.groupby(
            ["country", "tournament", "season", "week", "away_team"]
        ).last().reset_index()[[
            "country", "tournament", "season", "week",
            "away_team", "away_team_momentum_count", "away_team_momentum_value",
            "home_team", "home_team_momentum_count", "home_team_momentum_value"
        ]]

        home_last_momentum.rename(columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_team_momentum_count": "team_momentum_count",
            "home_team_momentum_value": "team_momentum_value",
            "away_team_momentum_count": "opponent_momentum_count",
            "away_team_momentum_value": "opponent_momentum_value"
        }, inplace=True)

        away_last_momentum.rename(columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_team_momentum_count": "team_momentum_count",
            "away_team_momentum_value": "team_momentum_value",
            "home_team_momentum_count": "opponent_momentum_count",
            "home_team_momentum_value": "opponent_momentum_value"
        }, inplace=True)

        final_momentum_df = pd.concat([home_last_momentum, away_last_momentum], ignore_index=True)

        if view_type == "Overall":
            final_summary_df = final_momentum_df.groupby(
                ["country", "tournament", "season", "team"]
            ).agg(
                total_team_momentum_count=("team_momentum_count", "sum"),
                total_team_momentum_value=("team_momentum_value", "sum"),
                total_opponent_momentum_count=("opponent_momentum_count", "sum"),
                total_opponent_momentum_value=("opponent_momentum_value", "sum")
            ).reset_index()
        elif view_type == "Weekly":
            final_summary_df = final_momentum_df.groupby(
                ["country", "tournament", "season", "week", "team"]
            ).agg(
                total_team_momentum_count=("team_momentum_count", "sum"),
                total_team_momentum_value=("team_momentum_value", "sum"),
                total_opponent_momentum_count=("opponent_momentum_count", "sum"),
                total_opponent_momentum_value=("opponent_momentum_value", "sum")
            ).reset_index()

        final_summary_df["team_momentum_per"] = final_summary_df["total_team_momentum_value"] / final_summary_df["total_team_momentum_count"]
        final_summary_df["opponent_momentum_per"] = final_summary_df["total_opponent_momentum_value"] / final_summary_df["total_opponent_momentum_count"]

        create_momentum_evolution_plot(final_summary_df, view_type)

    except Exception as e:
        st.error(f"No suitable data found.{e}")
        st.markdown(
            """
            <a href="https://github.com/urazakgul/datafc-web/issues" target="_blank" class="error-button">
                🛠️ Report Issue
            </a>
            """,
            unsafe_allow_html=True
        )