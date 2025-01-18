import os
import streamlit as st
from config import PLOT_STYLE
from code.utils.helpers import add_download_button, load_filtered_json_files, add_footer, turkish_upper, turkish_english_lower
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt

plt.style.use(PLOT_STYLE)

def create_player_heatmap_plot(filtered_hmap_data_df, league, season, league_display, season_display, team, last_round, player_name):

    pitch = VerticalPitch(
        pitch_type='opta',
        corner_arcs=True,
        half=False,
        label=False,
        tick=False
    )
    fig, ax = pitch.draw(figsize=(16, 16))

    pitch.kdeplot(
        filtered_hmap_data_df["x"],
        filtered_hmap_data_df["y"],
        ax=ax,
        fill=True,
        cmap="Reds",
        levels=100,
        alpha=0.6,
        zorder=0
    )

    ax.set_title(
        f"{league} {season} Sezonu Geçmiş {last_round} Haftada Oyuncu Isı Haritası",
        fontsize=16,
        fontweight="bold"
    )
    ax.text(
        0.5, 0.99,
        f"{turkish_upper(player_name)} ({turkish_upper(team)})",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        transform=ax.transAxes
    )

    add_footer(fig, x=0.5, y=0, ha="center")

    file_name = f"{league_display}_{season_display}_{last_round}_{turkish_english_lower(player_name)}_{turkish_english_lower(team)}_isi_haritasi.png"
    st.markdown(add_download_button(fig, file_name=file_name), unsafe_allow_html=True)
    st.pyplot(fig)

def main(league, season, league_display, season_display, team, player):
    try:

        directories = os.path.join(os.path.dirname(__file__), "../../data/sofascore/raw/")

        games_data = load_filtered_json_files(directories, "games", league_display, season_display)
        heat_maps_data = load_filtered_json_files(directories, "heat_maps", league_display, season_display)

        games_data = games_data[games_data["status"] == "Ended"]
        games_data = games_data[['game_id', 'tournament', 'season', 'round', 'home_team', 'away_team']]

        hmap_data_df = heat_maps_data.merge(
            games_data,
            on=['game_id', 'tournament', 'season', 'round'],
            how='left'
        )

        hmap_data_df['team_name'] = hmap_data_df.apply(
            lambda row: row['home_team'] if row['team'] == 'home' else row['away_team'],
            axis=1
        )

        filtered_hmap_data_df = hmap_data_df[
            (hmap_data_df['team_name'] == team) &
            (hmap_data_df['player_name'] == player)
        ]

        last_round = games_data['round'].max()

        create_player_heatmap_plot(filtered_hmap_data_df, league, season, league_display, season_display, team, last_round, player)

    except Exception as e:
        st.error("Uygun veri bulunamadı.")
        st.markdown(
            """
            <a href="https://github.com/urazakgul/buanalitikfutbol-app/issues" target="_blank" class="error-button">
                🛠️ Hata bildir
            </a>
            """,
            unsafe_allow_html=True
        )