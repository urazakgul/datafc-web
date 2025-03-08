import streamlit as st
import pandas as pd
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
from modules.homepage import get_data
from config import event_colors, PLOT_STYLE

plt.style.use(PLOT_STYLE)

def fill_team_name(df):
    df["team_name"] = df.groupby("id")["team_name"].transform(lambda x: x.ffill().bfill())
    return df

def fill_opponent_team_name(df):
    df["opponent_team_name"] = df.groupby("id")["opponent_team_name"].transform(lambda x: x.ffill().bfill())
    return df

def merge_match_data(match_data_df, shots_data_df):
    filtered_shots = shots_data_df[shots_data_df["shot_type"] == "goal"][
        ["tournament", "season", "week", "game_id", "player_name", "is_home", "goal_type", "xg"]
    ]
    merged_df = match_data_df.merge(filtered_shots, on=["tournament", "season", "week", "game_id"])
    return merged_df[~merged_df["goal_type"].isin(["penalty", "own"])]

def create_goal_network_plot(side_data, team, last_round, plot_type, side, combined_option):
    title_for = f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Goal Networks and Offensive Actions\n\n(Goals scored by {team})"
    title_against = f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Goal Networks and Opponent Offensive Actions\n\n(Goals conceded by {team})"

    title_jdp_for = f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Goal Networks and Offensive Actions Shaped by Juego de Posición\n\n(Goals scored by {team})"
    title_jdp_against = f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Goal Networks and Offensive Actions Shaped by Juego de Posición\n\n(Goals conceded by {team})"

    title = title_for if side == "For" else title_against
    title_jdp = title_jdp_for if side == "For" else title_jdp_against

    if plot_type == "Combined":
        if combined_option == "Network with Heatmap":
            pitch = VerticalPitch(
                pitch_type="opta",
                corner_arcs=True,
                half=False,
                label=False,
                tick=False
            )
            fig, ax = pitch.draw(figsize=(16, 16))

            kde_data = side_data[side_data["event_type"] != "goal"]
            pitch.kdeplot(
                kde_data["player_x"],
                kde_data["player_y"],
                ax=ax,
                fill=True,
                cmap="Reds",
                levels=100,
                alpha=0.6,
                zorder=0
            )

            for _, row in side_data.iterrows():
                color = event_colors.get(row["event_type"], "black")
                pitch.scatter(
                    row["player_x"],
                    row["player_y"],
                    ax=ax,
                    color=color,
                    s=50,
                    alpha=0.6,
                    edgecolors="black",
                    zorder=2
                )

            for _, group in side_data.groupby("id"):
                pitch.lines(
                    group["player_x"][:-1],
                    group["player_y"][:-1],
                    group["player_x"][1:],
                    group["player_y"][1:],
                    ax=ax,
                    lw=1,
                    color="blue",
                    alpha=0.2,
                    zorder=1
                )

            handles = [plt.Line2D([0], [0], marker="o", color=color, markersize=7, linestyle="None") for _, color in event_colors.items()]
            legend_labels = [label.capitalize() for label in event_colors.keys()]
            ax.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=3, fontsize=8)

            ax.set_title(title, fontsize=14, fontweight="bold", pad=35)
            fig.suptitle("Data: SofaScore, Prepared by @urazdev", y=0, x=0.5, fontsize=10, fontstyle="italic", color="gray")

        elif combined_option == "Juego de Posici\u00f3n":
            combined_data_filtered = side_data[side_data["event_type"] != "goal"]

            pitch = VerticalPitch(
                pitch_type="opta",
                line_zorder=2,
                corner_arcs=True,
                half=False,
                label=False,
                tick=False
            )
            fig, ax = pitch.draw(figsize=(16, 16))

            bin_statistic = pitch.bin_statistic_positional(
                combined_data_filtered["player_x"],
                combined_data_filtered["player_y"],
                statistic="count",
                positional="full",
                normalize=True
            )

            pitch.heatmap_positional(
                bin_statistic,
                ax=ax,
                cmap="Reds",
                edgecolors='#000'
            )

            labels = pitch.label_heatmap(
                bin_statistic,
                color='#000',
                fontsize=16,
                ax=ax,
                ha="center",
                va="center",
                str_format='{:.0%}'
            )

            ax.set_title(title_jdp, fontsize=12, fontweight="bold")
            fig.suptitle("Data: SofaScore, Prepared by @urazdev", y=0, x=0.5, fontsize=10, fontstyle="italic", color="gray")

    elif plot_type == "Separated":
        all_rounds = list(range(1, last_round + 1))
        existing_rounds = sorted(side_data["week"].unique())
        missing_rounds = set(all_rounds) - set(existing_rounds)

        for missing_round in missing_rounds:
            empty_data = pd.DataFrame({
                "week": [missing_round],
                "player_x": [None],
                "player_y": [None],
                "event_type": ["No Data"],
                "id": [None]
            })
            side_data = pd.concat([side_data, empty_data], ignore_index=True)

        rounds = sorted(side_data["week"].unique())

        n_cols = 5
        n_rows = -(-len(rounds) // n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 5))
        axes = axes.flatten()

        for i, round_num in enumerate(rounds):
            round_data = side_data[side_data["week"] == round_num]
            ax = axes[i]
            pitch = VerticalPitch(
                pitch_type="opta",
                corner_arcs=True,
                half=False,
                label=False,
                tick=False
            )
            pitch.draw(ax=ax)

            if not round_data.empty and round_data["event_type"].iloc[0] != "No Data":
                for _, row in round_data.iterrows():
                    color = event_colors.get(row["event_type"], "black")
                    pitch.scatter(
                        row["player_x"],
                        row["player_y"],
                        ax=ax,
                        color=color,
                        s=50,
                        alpha=0.6,
                        edgecolors="black",
                        zorder=2
                    )

                for _, group in round_data.groupby("id"):
                    pitch.lines(
                        group["player_x"][:-1],
                        group["player_y"][:-1],
                        group["player_x"][1:],
                        group["player_y"][1:],
                        ax=ax,
                        lw=1,
                        color="blue",
                        alpha=0.2,
                        zorder=1
                    )
            else:
                ax.text(0.5, 0.5, "No goal network found.", fontsize=10, ha="center", va="center", transform=ax.transAxes)

            ax.set_title(f"Week {round_num}", fontsize=10, fontweight="bold", pad=20)

        for j in range(len(rounds), len(axes)):
            axes[j].axis("off")

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.93)
        fig.text(0.5, 0.02, "Data: SofaScore, Prepared by @urazdev", ha="center", fontsize=14, fontstyle="italic", color="gray")

    st.pyplot(fig)

def main(team=None, plot_type=None, side=None, combined_option=None):
    try:

        match_data_df = get_data("match_data")
        shots_data_df = get_data("shots_data")
        goal_networks_data_df = get_data("goal_networks_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]
        match_data_df = match_data_df[["tournament", "season", "week", "game_id", "home_team", "away_team"]]
        match_shots_data_df = merge_match_data(match_data_df, shots_data_df)

        goal_networks_data_scored_df = goal_networks_data_df.copy()
        goal_networks_data_conceded_df = goal_networks_data_df.copy()

        goal_networks_data_scored_df["team_name"] = None
        goal_networks_data_conceded_df["opponent_team_name"] = None

        for game_id in match_shots_data_df["game_id"].unique():
            match_data = match_shots_data_df[match_shots_data_df["game_id"] == game_id]
            for _, row in match_data.iterrows():
                team_name = row["home_team"] if row["is_home"] else row["away_team"]
                opponent_team_name = row["away_team"] if row["is_home"] else row["home_team"]

                goal_networks_data_scored_df.loc[
                    (goal_networks_data_scored_df["game_id"] == game_id) &
                    (goal_networks_data_scored_df["player_name"] == row["player_name"]) &
                    (goal_networks_data_scored_df["event_type"] == "goal"), "team_name"
                ] = team_name

                goal_networks_data_conceded_df.loc[
                    (goal_networks_data_conceded_df["game_id"] == game_id) &
                    (goal_networks_data_conceded_df["player_name"] == row["player_name"]) &
                    (goal_networks_data_conceded_df["event_type"] == "goal"), "opponent_team_name"
                ] = opponent_team_name

        goal_networks_data_scored_df = fill_team_name(goal_networks_data_scored_df)
        goal_networks_data_conceded_df = fill_opponent_team_name(goal_networks_data_conceded_df)

        if side == "For":
            for _, group in goal_networks_data_scored_df.groupby("id"):
                if (group["event_type"] == "goal").any() and group.loc[group["event_type"] == "goal", "goal_shot_x"].iloc[0] != 100:
                    goal_networks_data_scored_df.loc[group.index, ["player_x", "player_y"]] = 100 - group[["player_x", "player_y"]]
        elif side == "Against":
            for _, group in goal_networks_data_conceded_df.groupby("id"):
                if (group["event_type"] == "goal").any() and group.loc[group["event_type"] == "goal", "goal_shot_x"].iloc[0] != 100:
                    goal_networks_data_conceded_df.loc[group.index, ["player_x", "player_y"]] = 100 - group[["player_x", "player_y"]]

        goal_networks_data_scored_df = goal_networks_data_scored_df.merge(match_data_df, on=["tournament", "season", "week", "game_id"])
        goal_networks_data_conceded_df = goal_networks_data_conceded_df.merge(match_data_df, on=["tournament", "season", "week", "game_id"])

        if side == "For":
            side_data = goal_networks_data_scored_df[goal_networks_data_scored_df["team_name"] == team]
        elif side == "Against":
            side_data = goal_networks_data_conceded_df[goal_networks_data_conceded_df["opponent_team_name"] == team]

        last_round = match_data_df.loc[(match_data_df["home_team"] == team) | (match_data_df["away_team"] == team), "week"].max()

        create_goal_network_plot(side_data, team, last_round, plot_type, side, combined_option)

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