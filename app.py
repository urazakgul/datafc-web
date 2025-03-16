import streamlit as st
from modules.homepage import display_homepage
from modules.team_based import display_team_based
from modules.team_comparison import display_team_comparison
from modules.player_based import display_player_based
from modules.player_comparison import display_player_comparison
from modules.match_comparison import display_match_comparison
from modules.analysis import display_eda_analysis, display_predictive_analytics
from code.utils.helpers import load_styles
from st_social_media_links import SocialMediaIcons
from streamlit_option_menu import option_menu

def configure_app():
    st.set_page_config(
        page_title="Data FC",
        page_icon=":soccer:",
        layout="wide"
    )
    load_styles()

def render_sidebar(social_media_links):
    with st.sidebar:
        # st.image("./imgs/datafc.PNG", use_container_width=True)

        social_media_icons = SocialMediaIcons(
            [link["url"] for link in social_media_links.values()],
            colors=[link["color"] for link in social_media_links.values()]
        )
        social_media_icons.render(sidebar=True)

        coffee_button = """
        <div style="text-align: center; margin-top: 20px;">
            <a href="https://www.buymeacoffee.com/urazdev" target="_blank">
                <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png"
                alt="Buy Me a Coffee"
                style="height: 50px; width: 217px;">
            </a>
        </div>
        """
        st.sidebar.markdown(coffee_button, unsafe_allow_html=True)
        st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

def main_menu():
    if "current_section" not in st.session_state:
        st.session_state["current_section"] = "Home"

    return option_menu(
        menu_title=None,
        options=["Home", "Team", "Player", "Match", "Analysis", "Metadata"],
        icons=["house", "shield", "person", "calendar", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#262730"},
            "icon": {"color": "#BABCBE", "font-size": "18px"},
            "nav-link": {
                "font-size": "20px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#3E4042",
                "color": "#BABCBE",
            },
            "nav-link-selected": {
                "background-color": "#262730",
                "color": "#fff",
            },
        },
    )

def handle_team_section():
    selection = st.sidebar.radio(
        "Team",
        ["Single Team", "Team Comparison"],
        index=None,
        label_visibility="hidden"
    )
    if selection == "Single Team":
        display_team_based()
    elif selection == "Team Comparison":
        display_team_comparison()

def handle_player_section():
    selection = st.sidebar.radio(
        "Player",
        ["Single Player", "Player Comparison"],
        index=None,
        label_visibility="hidden"
    )
    if selection == "Single Player":
        display_player_based()
    elif selection == "Player Comparison":
        display_player_comparison()

def handle_match_section():
    selection = st.sidebar.radio(
        "Match",
        ["Single Team", "Team Comparison"],
        index=None,
        label_visibility="hidden"
    )
    if selection == "Single Team":
        st.info("The Single Team section will be added soon.")
    elif selection == "Team Comparison":
        display_match_comparison()

def handle_analysis_section():
    selection = st.sidebar.radio(
        "Analysis",
        ["EDA", "Prediction"],
        index=None,
        label_visibility="hidden"
    )
    if selection == "EDA":
        display_eda_analysis()
    elif selection == "Prediction":
        display_predictive_analytics()

def run_app():
    configure_app()

    social_media_links = {
        "X": {"url": "https://www.x.com/urazdev", "color": "#fff"},
        "GitHub": {"url": "https://www.github.com/urazakgul", "color": "#fff"},
        "Reddit": {"url": "https://www.reddit.com/user/urazdev/", "color": "#fff"},
        "LinkedIn": {"url": "https://www.linkedin.com/in/uraz-akg%C3%BCl-439b36239/", "color": "#fff"},
    }
    render_sidebar(social_media_links)

    general_section = main_menu()
    st.session_state["current_section"] = general_section

    data_loaded = st.session_state.get("imported_data", None) is not None and bool(st.session_state["imported_data"])
    league_season_confirmed = st.session_state.get("league_season_confirmed", False)

    if general_section == "Home":
        display_homepage()
    elif general_section in ["Team", "Player", "Match", "Analysis"]:
        if league_season_confirmed and data_loaded:
            if general_section == "Team":
                handle_team_section()
            elif general_section == "Player":
                handle_player_section()
            elif general_section == "Match":
                handle_match_section()
            elif general_section == "Analysis":
                handle_analysis_section()
        else:
            st.warning("Please select the league and season from the homepage first, and make sure the data is loaded.")
    elif general_section == "Metadata":
        st.info("The Metadata section will be added soon.")

if __name__ == "__main__":
    run_app()