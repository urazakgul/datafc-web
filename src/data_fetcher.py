import unidecode
import re
from .supabase_client import get_supabase_client

_league_seasons_cache = None

def load_league_seasons_data():
    global _league_seasons_cache
    if _league_seasons_cache is None:
        supabase = get_supabase_client()
        data = supabase.table("league_seasons").select("*").execute()
        _league_seasons_cache = data.data if data.data else []
    return _league_seasons_cache

def get_countries():
    data = load_league_seasons_data()
    countries = {row["country"] for row in data}
    return sorted(countries)

def get_leagues_by_country(country=None):
    if not country:
        return []
    data = load_league_seasons_data()
    leagues = {row["league"] for row in data if row["country"] == country}
    return sorted(leagues)

def get_seasons_by_league_and_country(league=None, country=None):
    if not league or not country:
        return []
    data = load_league_seasons_data()
    seasons = {
        row["season"]
        for row in data
        if row["league"] == league and row["country"] == country
    }
    return sorted(seasons)

def get_teams_by_league_and_season(country, league, season):
    data = get_supabase_client()
    teams = data.table("teams").select("team_name").eq("country", country).eq("league", league).eq("season", season).order("team_name").execute()
    return [row["team_name"] for row in teams.data] if teams.data else []

def normalize_league_name(league):
    return (
        league.lower()
        .replace("ü", "u").replace("ö", "o").replace("ş", "s")
        .replace("ç", "c").replace("ı", "i").replace("ğ", "g")
        .replace(" ", "-")
    )

def normalize_season(season):
    return season.replace("/", "")

def normalize_team_name(team_name):
    name = unidecode.unidecode(team_name)
    name = name.lower().replace(" ", "_")
    name = re.sub(r'\W+', '', name)
    return name

def get_team_logo_url(team_name, league, season):
    supabase = get_supabase_client()
    folder = f"{normalize_league_name(league)}_{normalize_season(season)}"
    filename = f"{normalize_team_name(team_name)}.png"
    path = f"{folder}/{filename}"
    url = supabase.storage.from_("team-logos").get_public_url(path)
    return url