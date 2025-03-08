import streamlit as st
from mplsoccer import VerticalPitch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from modules.homepage import get_data
from config import PLOT_STYLE, shot_colors

plt.style.use(PLOT_STYLE)

def create_player_shot_location_plot(df_goals, df_no_goals, selected_team, selected_player, selected_situation=None, show_xg_based=None, include_shot_type=None):
    pitch = VerticalPitch(
        pitch_type="opta",
        corner_arcs=True,
        half=False,
        label=False,
        tick=False
    )

    fig, ax = pitch.draw(figsize=(16, 16))

    if show_xg_based == "xG-Adjusted View":
        df_goals["size"] = df_goals["xg"] * 300
        df_no_goals["size"] = df_no_goals["xg"] * 300
        view_title = " (xG-Adjusted)"
    else:
        df_goals["size"] = 150
        df_no_goals["size"] = 150
        view_title = ""

    if include_shot_type == "Break by Shot Type":
        for shot_type, color in shot_colors.items():
            shot_subset = df_no_goals[df_no_goals["shot_type"] == shot_type]
            if not shot_subset.empty:
                pitch.scatter(
                    shot_subset["player_coordinates_x"],
                    shot_subset["player_coordinates_y"],
                    edgecolors="black",
                    c=color,
                    marker="h",
                    alpha=0.6,
                    s=shot_subset["size"],
                    label=f"{shot_type.capitalize()}",
                    ax=ax
                )
    else:
        pitch.scatter(
            df_no_goals["player_coordinates_x"],
            df_no_goals["player_coordinates_y"],
            edgecolors="black",
            c="gray",
            marker="h",
            alpha=0.3,
            s=df_no_goals["size"],
            label="No Goal",
            ax=ax
        )

    pitch.scatter(
        df_goals["player_coordinates_x"],
        df_goals["player_coordinates_y"],
        edgecolors="black",
        c="red",
        marker="h",
        alpha=0.7,
        s=df_goals["size"],
        label="Goal",
        ax=ax
    )

    total_shots = len(df_goals) + len(df_no_goals)
    conversion_rate = len(df_goals) / total_shots * 100 if total_shots > 0 else 0

    non_penalty_goals = df_goals[df_goals["situation"] != "penalty"]
    non_penalty_shots = total_shots - len(df_goals[df_goals["situation"] == "penalty"])
    non_penalty_conversion_rate = len(non_penalty_goals) / non_penalty_shots * 100 if non_penalty_shots > 0 else 0

    title_suffix = f" ({selected_situation})" if selected_situation else ""
    ax.text(
        0.5, 1.05,
        s=(f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Shot Locations for {selected_player} ({selected_team}){view_title}"),
        size=14,
        fontweight="bold",
        va="center",
        ha="center",
        transform=ax.transAxes
    )

    if selected_situation == "All":
        ax.text(
            0.5, 1.025,
            s=(f"Shot Conversion Rate{title_suffix}: %{conversion_rate:.1f} (Including Penalties), %{non_penalty_conversion_rate:.1f} (Excluding Penalties)"),
            size=10,
            fontstyle="italic",
            va="center",
            ha="center",
            transform=ax.transAxes
        )
    else:
        ax.text(
            0.5, 1.025,
            s=(f"Shot Conversion Rate{title_suffix}: %{conversion_rate:.1f}"),
            size=10,
            fontstyle="italic",
            va="center",
            ha="center",
            transform=ax.transAxes
        )

    ax.text(
        0.5, -0.01,
        s="Data: SofaScore, Prepared by @urazdev\nOwn goals excluded.",
        size=10,
        color=pitch.line_color,
        fontstyle="italic",
        va="center",
        ha="center",
        transform=ax.transAxes
    )

    legend_elements = [
        Line2D([0], [0], marker="h", color="none", label="Goal", markerfacecolor="red", markersize=10, markeredgecolor="black")
    ]

    if include_shot_type == "Break by Shot Type":
        for shot_type, color in shot_colors.items():
            legend_elements.append(
                Line2D([0], [0], marker="h", color="none", label=shot_type.capitalize(), markerfacecolor=color, markersize=10, markeredgecolor="black")
            )
    else:
        legend_elements.append(
            Line2D([0], [0], marker="h", color="none", label="No Goal", markerfacecolor="gray", markersize=10, markeredgecolor="black")
        )

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5 if include_shot_type == "Break by Shot Type" else 2,
        fontsize=12,
        frameon=False,
        facecolor="white",
        edgecolor="black",
        handletextpad=0.2
    )

    st.pyplot(fig)

def main(selected_team=None, selected_player=None, selected_situation=None, show_xg_based=None, include_shot_type=None):
    try:

        match_data_df = get_data("match_data")
        shots_data_df = get_data("shots_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]
        match_data_df = match_data_df[["season", "week", "game_id", "home_team", "away_team"]]

        shots_data_df = shots_data_df[[
            "season", "week", "game_id", "player_name", "is_home", "shot_type", "situation",
            "goal_mouth_location", "player_coordinates_x", "player_coordinates_y", "xg"
        ]]

        shots_data_df = shots_data_df.merge(
            match_data_df,
            on=["season", "week", "game_id"],
            how="left"
        )

        shots_data_df["team_name"] = shots_data_df.apply(
            lambda row: row["home_team"] if row["is_home"] else row["away_team"], axis=1
        )

        shots_data_df["is_goal"] = shots_data_df["shot_type"].apply(lambda x: 1 if x == "goal" else 0)
        shots_data_df["player_coordinates_x"] = 100 - shots_data_df["player_coordinates_x"]
        shots_data_df["player_coordinates_y"] = 100 - shots_data_df["player_coordinates_y"]

        player_data = shots_data_df[(shots_data_df["team_name"] == selected_team) & (shots_data_df["player_name"] == selected_player)]

        df_goals = player_data[player_data["is_goal"] == 1]
        df_no_goals = player_data[player_data["is_goal"] == 0]

        create_player_shot_location_plot(df_goals, df_no_goals, selected_team, selected_player, selected_situation, show_xg_based, include_shot_type)

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