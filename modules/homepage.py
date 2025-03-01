import os
import glob
import gzip
import pandas as pd
import streamlit as st

def initialize_session_state():
    if "league_season_confirmed" not in st.session_state:
        st.session_state["league_season_confirmed"] = False
    if "selected_league" not in st.session_state:
        st.session_state["selected_league"] = None
    if "selected_season" not in st.session_state:
        st.session_state["selected_season"] = None
    if "selected_league_original" not in st.session_state:
        st.session_state["selected_league_original"] = None
    if "selected_season_original" not in st.session_state:
        st.session_state["selected_season_original"] = None
    if "selected_country" not in st.session_state:
        st.session_state["selected_country"] = None

def render_welcome_message():
    st.markdown('<h1 class="big-font">The Data FC Web Application</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div>
            <p>
            <b>Data FC</b> is a cutting-edge web application that brings a <mark>data-driven approach</mark> to football analysis, delivering <b>high-quality visualizations</b> and <b>in-depth insights</b> for the <b>Premier League</b> and <b>Süper Lig</b>.
            Through a <mark>comprehensive analytical framework</mark>, it enhances the understanding of the game for enthusiasts while providing professionals with <mark>valuable tools</mark> that support <b>strategic decision-making</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

def render_league_season_selection():
    st.markdown("<h3>League and Season Selection</h3>", unsafe_allow_html=True)
    st.markdown("""
        <div style="font-style: italic; color: gray;">
            Please select the league and season you want to analyze. Once you save your selection, the analyses will be performed based on this information.
            If you wish to change your selection, you can return to this page at any time.
        </div>
    """, unsafe_allow_html=True)

    leagues = ["Premier League", "Süper Lig"]
    seasons = ["2024/25"]

    selected_league = st.selectbox(
        "League:",
        leagues,
        index=None,
        key="league_select",
        placeholder="Leagues"
    )
    selected_season = st.selectbox(
        "Season:",
        seasons,
        index=None,
        key="season_select",
        placeholder="Seasons"
    )

    if st.button("Save"):
        league_mapping = {
            "Premier League": ("england", "premier_league"),
            "Süper Lig": ("turkey", "super_lig")
        }
        season_mapping = {"2024/25": "2425"}

        country, league = league_mapping.get(selected_league, (None, None))
        selected_season_code = season_mapping.get(selected_season, selected_season)

        if country and league:
            st.session_state["selected_league_original"] = selected_league
            st.session_state["selected_season_original"] = selected_season
            st.session_state["selected_country"] = country
            st.session_state["selected_league"] = league
            st.session_state["selected_season"] = selected_season_code
            st.session_state["league_season_confirmed"] = True
            st.success(f"Data for the {selected_league} {selected_season} season is being loaded. Please wait.")

            import_data("./data/sofascore/raw", country, league, selected_season_code)

def load_filtered_json_files(directory: str, country: str, league: str, season: str, subdirectory: str) -> pd.DataFrame:
    path = os.path.join(directory, subdirectory, f"sofascore_{country}_{league}_{season}_{subdirectory}.json.gz")
    files = glob.glob(path)

    dataframes = []
    for file in files:
        with gzip.open(file, "rt", encoding="utf-8") as f:
            dataframes.append(pd.read_json(f))

    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

def import_data(directory: str, country: str, league: str, season: str):
    subdirectories = [
        "coordinates_data",
        "goal_networks_data",
        "lineups_data",
        "match_data",
        "match_odds_data",
        "match_stats_data",
        "momentum_data",
        "shots_data",
        "standings_data",
        "substitutions_data"
    ]

    st.session_state["imported_data"] = {}

    with st.spinner("Loading data..."):
        for subdir in subdirectories:
            df = load_filtered_json_files(directory, country, league, season, subdir)
            st.session_state["imported_data"][subdir] = df

        st.session_state["league_season_confirmed"] = True
        st.success("Data has been successfully loaded.")

def get_data(data_type: str) -> pd.DataFrame:
    if "imported_data" not in st.session_state:
        st.error("Please select a league and season from the homepage and wait for the data to be loaded.")
        st.stop()

    return st.session_state["imported_data"].get(data_type, pd.DataFrame())

def display_homepage():
    initialize_session_state()
    render_welcome_message()
    render_league_season_selection()

if __name__ == "__main__":
    display_homepage()