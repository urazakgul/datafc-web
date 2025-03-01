import streamlit as st
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_player_heatmap_plot(filtered_coord_data_df, team, player_name):

    pitch = VerticalPitch(
        pitch_type="opta",
        corner_arcs=True,
        half=False,
        label=False,
        tick=False
    )
    fig, ax = pitch.draw(figsize=(16, 16))

    pitch.kdeplot(
        filtered_coord_data_df["x"],
        filtered_coord_data_df["y"],
        ax=ax,
        fill=True,
        cmap="Reds",
        levels=100,
        alpha=0.6,
        zorder=0
    )

    ax.set_title(
        f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Player Heatmap",
        fontsize=16,
        fontweight="bold"
    )
    ax.text(
        0.5, 0.99,
        f"{player_name} ({team})",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        transform=ax.transAxes
    )

    add_footer(fig, x=0.5, y=0, ha="center")

    st.pyplot(fig)

def main(team, player):
    try:

        match_data_df = get_data("match_data")
        coordinates_data_df = get_data("coordinates_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]
        match_data_df = match_data_df[['game_id', 'tournament', 'season', 'week', 'home_team', 'away_team']]

        coord_data_df = coordinates_data_df.merge(
            match_data_df,
            on=['game_id', 'tournament', 'season', 'week'],
            how='left'
        )

        coord_data_df['team_name'] = coord_data_df.apply(
            lambda row: row['home_team'] if row['team'] == 'home' else row['away_team'],
            axis=1
        )

        filtered_coord_data_df = coord_data_df[
            (coord_data_df['team_name'] == team) &
            (coord_data_df['player_name'] == player)
        ]

        create_player_heatmap_plot(filtered_coord_data_df, team, player)

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