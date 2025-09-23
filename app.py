import streamlit as st
import pandas as pd
import importlib
from src.supabase_client import get_supabase_client
from src.data_fetcher import (
    get_countries,
    get_leagues_by_country,
    get_seasons_by_league_and_country,
    get_teams_by_league_and_season,
    get_team_logo_url
)
from src.components.html_snippets import (
    main_page_header,
    support_button,
    page_styles,
    team_analysis_header,
    prediction_header
)

st.set_page_config(page_title="DATAFC-WEB", page_icon="⚽", layout="centered")

st.markdown(page_styles(), unsafe_allow_html=True)

for k in ["page", "selected_country", "selected_league", "selected_season", "selected_action", "show_error", "is_loading"]:
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.page is None:
    st.session_state.page = "selection"

def fetch_all_data(table_name, filters=None, batch_size=1000):
    supabase = get_supabase_client()
    all_rows = []
    last_id = None

    while True:
        q = (
            supabase.table(table_name)
            .select("*")
            .order("row_id")
            .limit(batch_size)
        )
        if filters:
            for col, val in filters.items():
                q = q.eq(col, val)

        if last_id is not None:
            q = q.gt("row_id", last_id)

        res = q.execute()
        batch = res.data or []
        if not batch:
            break

        all_rows.extend(batch)
        last_id = batch[-1]["row_id"]

        if len(batch) < batch_size:
            break

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = (
            df.drop_duplicates(subset=["row_id"])
              .sort_values("row_id", kind="mergesort")
              .reset_index(drop=True)
        )
    return df

def fetch_historical_tff_matches(selected_tournament, selected_season, batch_size=1000):
    supabase = get_supabase_client()

    seasons = set()
    start = 0
    while True:
        res = (
            supabase.table("tff_historical_matches")
            .select("season, row_id")
            .order("row_id")
            .range(start, start + batch_size - 1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            break
        for item in rows:
            if item.get("season") is not None:
                seasons.add(item["season"])
        if len(rows) < batch_size:
            break
        start += batch_size

    all_seasons = sorted(seasons)

    def season_key(s):
        try:
            return int(str(s).split('/')[0])
        except Exception:
            return 0

    all_seasons = sorted(all_seasons, key=season_key)
    selected_idx = all_seasons.index(selected_season) if selected_season in all_seasons else len(all_seasons)
    last_seasons = all_seasons[max(0, selected_idx-3):selected_idx]
    if selected_season in all_seasons:
        last_seasons.append(selected_season)
    last_seasons = list(dict.fromkeys(last_seasons))

    all_rows = []

    for season in last_seasons:
        last_id = None
        while True:
            q = (
                supabase.table("tff_historical_matches")
                .select("*")
                .eq("season", season)
                .order("row_id")
                .limit(batch_size)
            )
            if last_id is not None:
                q = q.gt("row_id", last_id)

            if selected_tournament and season == selected_season:
                q = q.eq("tournament", selected_tournament)

            res = q.execute()
            batch = res.data or []
            if not batch:
                break

            all_rows.extend(batch)
            last_id = batch[-1]["row_id"]

            if len(batch) < batch_size:
                break

    df = pd.DataFrame(all_rows)

    if not df.empty:
        df = (
            df.drop_duplicates(subset=["row_id"])
              .sort_values(["season", "row_id"], kind="mergesort")
              .reset_index(drop=True)
        )

    return df

def selection_page():
    st.markdown(main_page_header(), unsafe_allow_html=True)
    st.markdown(support_button(), unsafe_allow_html=True)

    countries = get_countries()
    st.selectbox(
        "Country",
        countries,
        index=countries.index(st.session_state.selected_country) if st.session_state.selected_country in countries else None,
        placeholder="Please select a country",
        key="selected_country"
    )

    leagues = get_leagues_by_country(st.session_state.selected_country) if st.session_state.selected_country else []
    st.selectbox(
        "League",
        leagues,
        index=leagues.index(st.session_state.selected_league) if st.session_state.selected_league in leagues else None,
        placeholder="Please select a league",
        key="selected_league"
    )

    seasons = get_seasons_by_league_and_country(
        st.session_state.selected_league,
        st.session_state.selected_country
    ) if st.session_state.selected_league and st.session_state.selected_country else []
    st.selectbox(
        "Season",
        seasons,
        index=seasons.index(st.session_state.selected_season) if st.session_state.selected_season in seasons else None,
        placeholder="Please select a season",
        key="selected_season"
    )

    actions = ["Team Analysis", "Prediction"]
    actions_disabled = not (
        st.session_state.selected_country
        and st.session_state.selected_league
        and st.session_state.selected_season
    )

    if actions_disabled and st.session_state.selected_action is not None:
        st.session_state.selected_action = None

    st.selectbox(
        "What would you like to do?",
        actions,
        index=actions.index(st.session_state.selected_action) if st.session_state.selected_action in actions else None,
        placeholder="Please select an action",
        key="selected_action",
        disabled=actions_disabled
    )

    can_continue = all([
        st.session_state.selected_country,
        st.session_state.selected_league,
        st.session_state.selected_season,
        st.session_state.selected_action
    ])
    continue_disabled = (not can_continue) or bool(st.session_state.is_loading)

    if st.button("Continue", key="main_continue", disabled=continue_disabled):
        st.session_state.is_loading = True
        st.rerun()

    if st.session_state.is_loading:
        with st.spinner("Loading data, please wait..."):
            try:
                filters = {
                    "season": st.session_state.selected_season,
                    "tournament": st.session_state.selected_league,
                    "country": st.session_state.selected_country
                }

                if st.session_state.selected_action == "Team Analysis":
                    st.session_state['match_data'] = fetch_all_data("match_data", filters)
                    st.session_state['shots_data'] = fetch_all_data("shots_data", filters)
                    st.session_state['goal_networks_data'] = fetch_all_data("goal_networks_data", filters)
                    st.session_state['standings_data'] = fetch_all_data("standings_data", filters)
                    st.session_state['match_stats_data'] = fetch_all_data("match_stats_data", filters)
                    st.session_state['coordinates_data'] = fetch_all_data("coordinates_data", filters)
                    st.session_state['lineups_data'] = fetch_all_data("lineups_data", filters)
                    st.session_state['substitutions_data'] = fetch_all_data("substitutions_data", filters)
                    st.session_state['momentum_data'] = fetch_all_data("momentum_data", filters)

                    teams = get_teams_by_league_and_season(
                        st.session_state.selected_country,
                        st.session_state.selected_league,
                        st.session_state.selected_season
                    )
                    team_logo_urls = {}
                    for team in teams:
                        logo_url = get_team_logo_url(
                            team,
                            st.session_state.selected_league,
                            st.session_state.selected_season
                        )
                        team_logo_urls[team] = logo_url
                    st.session_state['team_logo_urls'] = team_logo_urls

                    st.session_state.page = "team_analysis"

                elif st.session_state.selected_action == "Prediction":
                    if st.session_state.selected_country == "Turkey":
                        st.session_state['tff_historical_matches'] = fetch_historical_tff_matches(
                            st.session_state.selected_league,
                            st.session_state.selected_season
                        )

                    st.session_state['match_data'] = fetch_all_data("match_data", filters)

                    teams = get_teams_by_league_and_season(
                        st.session_state.selected_country,
                        st.session_state.selected_league,
                        st.session_state.selected_season
                    )
                    st.session_state['teams'] = teams

                    st.session_state.page = "prediction"

                st.session_state.is_loading = False
                st.rerun()

            except Exception as e:
                st.session_state.is_loading = False
                st.error(f"Data loading failed: {e}")

    if st.session_state.show_error:
        st.error("Please fill in all selections before continuing!")
        st.session_state.show_error = False

def team_analysis_page():
    st.markdown(team_analysis_header(), unsafe_allow_html=True)

    if st.button("Go to Main Page", key="go_main_page_team_analysis"):
        with st.spinner("Returning to main page..."):
            st.session_state.page = 'selection'
            st.session_state.selected_country = None
            st.session_state.selected_league = None
            st.session_state.selected_season = None
            st.session_state.selected_action = None
            st.session_state.ta_selected_team = None
            st.session_state.ta_selected_team_analysis = None
            st.session_state.ta_scope = None
            st.session_state.is_loading = False
            st.rerun()

    st.selectbox(
        "Country",
        [st.session_state.selected_country] if st.session_state.selected_country else [],
        index=0 if st.session_state.selected_country else None,
        key="selected_country",
        disabled=True
    )
    st.selectbox(
        "League",
        [st.session_state.selected_league] if st.session_state.selected_league else [],
        index=0 if st.session_state.selected_league else None,
        key="selected_league",
        disabled=True
    )
    st.selectbox(
        "Season",
        [st.session_state.selected_season] if st.session_state.selected_season else [],
        index=0 if st.session_state.selected_season else None,
        key="selected_season",
        disabled=True
    )

    scope_options = ["Team-specific", "Comparative"]
    prev_scope = st.session_state.get("ta_scope")
    selected_scope = st.selectbox(
        "Scope",
        scope_options,
        index=(scope_options.index(prev_scope) if prev_scope in scope_options else None),
        placeholder="Please select a scope",
        key="ta_scope"
    )

    if selected_scope != prev_scope:
        st.session_state.ta_selected_team = None
        st.session_state.ta_selected_team_analysis = None

    if selected_scope is None:
        return

    all_analysis_options = {
        "Shot Locations": "shot_location",
        "Goal Networks": "goal_network",
        "Actual vs Expected Goal Differences": "actual_expected_gd",
        "Expected Points": "expected_points",
        "Finishing Over/Underperformance Trend": "fin_perf_trend",
        "On-Target vs Off-Target xG per Shot": "xg_per_shot_on_vs_off",
        "Aggregated Match Stats": "agg_match_stats",
        "Possession vs Touches in Box": "possession_vs_touches_in_box",
        "Ground Duels vs Aerial Duels": "ground_vs_aerial_duels",
        "Similarity": "similarity",
        "Goal Breakdown": "goal_breakdown",
        "Expected Goals by Situation": "xg_by_situation",
        "Goals vs Expected Goals Difference by Situation": "goals_vs_xg_by_situation",
        "Momentum": "momentum",
        "Geometry": "geometry",
        "Percentage of Total Time Spent in Each Match State": "time_state_percentages",
        "Substitution Impact on Momentum": "subs_momentum_impact",
        "Player Analysis": "player_analysis",
    }
    analyses_requiring_team = {
        "Shot Locations",
        "Goal Networks",
        "Finishing Over/Underperformance Trend",
        "Similarity",
        "Goal Breakdown",
        "Player Analysis",
    }

    if selected_scope == "Team-specific":
        analysis_options = {k: v for k, v in all_analysis_options.items() if k in analyses_requiring_team}
    else:
        analysis_options = {k: v for k, v in all_analysis_options.items() if k not in analyses_requiring_team}

    prev_analysis = st.session_state.get("ta_selected_team_analysis_prev")
    analysis_keys = list(analysis_options.keys())
    selected_analysis = st.selectbox(
        "Analysis",
        analysis_keys,
        index=(analysis_keys.index(st.session_state.ta_selected_team_analysis)
               if st.session_state.get("ta_selected_team_analysis") in analysis_keys else None),
        placeholder="Please select an analysis",
        key="ta_selected_team_analysis"
    )

    if selected_analysis != prev_analysis:
        st.session_state.ta_selected_team = None
    st.session_state.ta_selected_team_analysis_prev = selected_analysis

    selected_team = None
    if selected_scope == "Team-specific" and selected_analysis:
        teams = get_teams_by_league_and_season(
            st.session_state.selected_country,
            st.session_state.selected_league,
            st.session_state.selected_season
        )
        selected_team = st.selectbox(
            "Team",
            teams,
            index=(teams.index(st.session_state.ta_selected_team)
                   if st.session_state.get("ta_selected_team") in teams else None),
            placeholder="Please select a team",
            key="ta_selected_team"
        )

    if selected_analysis:
        module_name = f"src.analyses.team_analysis.{analysis_options[selected_analysis]}"
        try:
            analysis_module = importlib.import_module(module_name)
            if selected_scope == "Team-specific" and selected_team is not None:
                analysis_module.run(
                    team=selected_team,
                    country=st.session_state.selected_country,
                    league=st.session_state.selected_league,
                    season=st.session_state.selected_season
                )
            elif selected_scope == "Comparative":
                analysis_module.run(
                    country=st.session_state.selected_country,
                    league=st.session_state.selected_league,
                    season=st.session_state.selected_season
                )
        except ModuleNotFoundError:
            st.error(f"Module `{module_name}` not found!")
        except AttributeError:
            st.error(f"`run()` function not found in `{module_name}`!")
        except Exception as e:
            st.error(f"Error: {e}")

def prediction_page():
    st.markdown(prediction_header(), unsafe_allow_html=True)

    if st.button("Go to Main Page", key="go_main_page_prediction"):
        with st.spinner("Returning to main page..."):
            st.session_state.page = 'selection'
            st.session_state.selected_country = None
            st.session_state.selected_league = None
            st.session_state.selected_season = None
            st.session_state.selected_action = None
            st.session_state.is_loading = False
            st.rerun()

    st.selectbox(
        "Country",
        [st.session_state.selected_country] if st.session_state.selected_country else [],
        index=0 if st.session_state.selected_country else None,
        key="selected_country",
        disabled=True
    )
    st.selectbox(
        "League",
        [st.session_state.selected_league] if st.session_state.selected_league else [],
        index=0 if st.session_state.selected_league else None,
        key="selected_league",
        disabled=True
    )
    st.selectbox(
        "Season",
        [st.session_state.selected_season] if st.session_state.selected_season else [],
        index=0 if st.session_state.selected_season else None,
        key="selected_season",
        disabled=True
    )

    # analysis_modes = ["Historical Goal Performance", "Prediction", "Backtest"]
    analysis_modes = ["Historical Goal Performance", "Prediction"]
    selected_analysis_mode = st.selectbox(
        "Mode",
        analysis_modes,
        index=None,
        placeholder="Please select a mode",
        key="mode_selection"
    )

    if selected_analysis_mode == "Historical Goal Performance":
        module_name = "src.analyses.prediction.hist_goal_perf"
        try:
            analysis_module = importlib.import_module(module_name)
            analysis_module.run(
                country=st.session_state.selected_country,
                league=st.session_state.selected_league,
                season=st.session_state.selected_season,
            )
        except ModuleNotFoundError:
            st.warning(f"`{module_name}` not found. Please create this module with a `run(...)` entrypoint.")
        except AttributeError:
            st.error(f"`run()` function not found in `{module_name}`!")
        except Exception as e:
            st.error(f"Error: {e}")

    elif selected_analysis_mode == "Prediction":
        analysis_options = {
            "Dixon-Coles": "dixon_coles",
            "Bivariate Poisson": "bivariate_poisson",
            "Skellam Distribution": "skellam_distribution",
        }
        selected_prediction_method = st.selectbox(
            "Prediction Method",
            list(analysis_options.keys()),
            index=None,
            placeholder="Please select a prediction method",
            key="prediction_method"
        )

        if selected_prediction_method:
            module_name = f"src.analyses.prediction.{analysis_options[selected_prediction_method]}"
            try:
                analysis_module = importlib.import_module(module_name)
                analysis_module.run(
                    country=st.session_state.selected_country,
                    league=st.session_state.selected_league,
                    season=st.session_state.selected_season,
                )
            except ModuleNotFoundError:
                st.error(f"Module `{module_name}` not found!")
            except AttributeError:
                st.error(f"`run()` function not found in `{module_name}`!")
            except Exception as e:
                st.error(f"Error: {e}")

    # elif selected_analysis_mode == "Backtest":
    #     module_name = "src.analyses.prediction.backtest"
    #     try:
    #         analysis_module = importlib.import_module(module_name)
    #         analysis_module.run(
    #             country=st.session_state.selected_country,
    #             league=st.session_state.selected_league,
    #             season=st.session_state.selected_season,
    #         )
    #     except ModuleNotFoundError:
    #         st.warning(f"`{module_name}` not found. Please create this module with a `run(...)` entrypoint.")
    #     except AttributeError:
    #         st.error(f"`run()` function not found in `{module_name}`!")
    #     except Exception as e:
    #         st.error(f"Error: {e}")

if st.session_state.page == "selection":
    selection_page()
elif st.session_state.page == "team_analysis":
    team_analysis_page()
elif st.session_state.page == "prediction":
    prediction_page()