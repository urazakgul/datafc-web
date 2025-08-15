import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from src.utils.session_data import (
    require_session_data,
    filter_matches_by_status
)

STAT_CONFIG = {
    "Ball Possession": {
        "stats": ["Ball possession"],
        "type": "mean",
        "xlabel": "Ball Possession (%)",
        "title": "Average Ball Possession",
        "ylim": (0, 100),
    },
    "Pass Success Rate": {
        "stats": ["Accurate passes", "Passes"],
        "type": "ratio",
        "xlabel": "Pass Success Rate (%)",
        "title": "Pass Success Rate",
        "ylim": (0, 100),
    },
    "Big Chances Conversion Rate": {
        "stats": ["Big chances scored", "Big chances missed"],
        "type": "ratio_2col",
        "xlabel": "Big Chances Conversion Rate (%)",
        "title": "Big Chances Conversion Rate",
        "ylim": (0, 100),
    },
    "Shot Accuracy": {
        "stats": ["Shots on target", "Total shots"],
        "type": "ratio",
        "xlabel": "Shot Accuracy (%)",
        "title": "Shot Accuracy",
        "ylim": (0, 100),
    },
    "Shot Ratio Inside/Outside Box": {
        "stats": ["Shots inside box", "Shots outside box"],
        "type": "custom_shot_ratio",
        "xlabel": "Inside/Outside Box Shot Ratio",
        "title": "Shot Ratio Inside/Outside Box",
    },
    "Touches in Penalty Area": {
        "stats": ["Touches in penalty area"],
        "type": "sum",
        "xlabel": "Touches in Penalty Area",
        "title": "Touches in Penalty Area",
    },
    "Final Third Entries": {
        "stats": ["Final third entries"],
        "type": "sum",
        "xlabel": "Final Third Entries",
        "title": "Final Third Entries",
    },
    "Successful Actions in Final Third": {
        "stats": ["Final third phase"],
        "type": "success_total_ratio",
        "xlabel": "Successful Actions in Final Third (%)",
        "title": "Successful Actions in Final Third (%)",
        "ylim": (0, 100),
    },
    "Difference Between Fouls Committed and Suffered": {
        "stats": ["Fouls"],
        "type": "foul_difference",
        "xlabel": "Difference (Committed - Suffered)",
        "title": "Difference Between Fouls Committed and Suffered",
    },
    "Cards Per Foul": {
        "stats": ["Yellow cards", "Red cards", "Fouls"],
        "type": "cards_per_foul",
        "xlabel": "Cards Per Foul",
        "title": "Cards Per Foul",
    },
    "Accurate Long Pass Rate": {
        "stats": ["Long balls"],
        "type": "success_total_ratio",
        "xlabel": "Accurate Long Pass Rate (%)",
        "title": "Accurate Long Pass Rate",
        "ylim": (0, 100),
    },
    "Accurate Cross Rate": {
        "stats": ["Crosses"],
        "type": "success_total_ratio",
        "xlabel": "Accurate Cross Rate (%)",
        "title": "Accurate Cross Rate",
        "ylim": (0, 100),
    },
}

def _clean_percent_columns(dataframe, columns_to_check, target_columns):
    for index, row in dataframe.iterrows():
        if any(keyword in row["stat_name"] for keyword in columns_to_check):
            for col in target_columns:
                dataframe.at[index, col] = row[col].replace("%", "").strip()
    return dataframe

def _clean_parenthesis_columns(dataframe, columns_to_check, target_columns):
    for index, row in dataframe.iterrows():
        if any(keyword in row["stat_name"] for keyword in columns_to_check):
            for col in target_columns:
                if "(" in row[col]:
                    dataframe.at[index, col] = row[col].split("(")[0].strip()
    return dataframe

def _compute_stat(df, config, master_df=None):
    t = config["type"]
    if t == "mean":
        stat_df = df[df["stat_name"] == config["stats"][0]].copy()
        stat_df["stat_value"] = pd.to_numeric(stat_df["stat_value"], errors="coerce")
        team_grouped = stat_df.groupby("team_name", as_index=False)["stat_value"].mean()
        team_grouped = team_grouped.rename(columns={"stat_value": config["xlabel"]})
        team_grouped[config["xlabel"]] = team_grouped[config["xlabel"]].round(2)
        team_grouped = team_grouped.sort_values(config["xlabel"], ascending=False)
        return team_grouped, config["xlabel"]
    elif t == "sum":
        stat_df = df[df["stat_name"] == config["stats"][0]].copy()
        stat_df["stat_value"] = pd.to_numeric(stat_df["stat_value"], errors="coerce")
        team_grouped = stat_df.groupby("team_name", as_index=False)["stat_value"].sum()
        team_grouped = team_grouped.rename(columns={"stat_value": config["xlabel"]})
        team_grouped = team_grouped.sort_values(config["xlabel"], ascending=False)
        return team_grouped, config["xlabel"]
    elif t == "ratio":
        if config.get("title") == "Shot Accuracy":
            acc = df[df["stat_name"] == config["stats"][0]].copy().drop_duplicates()
            tot = df[df["stat_name"] == config["stats"][1]].copy().drop_duplicates()
        else:
            acc = df[df["stat_name"] == config["stats"][0]].copy()
            tot = df[df["stat_name"] == config["stats"][1]].copy()
        acc["stat_value"] = pd.to_numeric(acc["stat_value"], errors="coerce")
        tot["stat_value"] = pd.to_numeric(tot["stat_value"], errors="coerce")
        a_team = acc.groupby("team_name", as_index=False)["stat_value"].sum()
        t_team = tot.groupby("team_name", as_index=False)["stat_value"].sum()
        merged = a_team.merge(t_team, on="team_name", suffixes=("_acc", "_tot"))
        merged[config["xlabel"]] = 100 * merged["stat_value_acc"] / merged["stat_value_tot"].replace(0, pd.NA)
        merged[config["xlabel"]] = merged[config["xlabel"]].round(2)
        merged = merged.sort_values(config["xlabel"], ascending=False)
        return merged, config["xlabel"]
    elif t == "ratio_2col":
        s = df[df["stat_name"] == config["stats"][0]].copy()
        m = df[df["stat_name"] == config["stats"][1]].copy()
        s["stat_value"] = pd.to_numeric(s["stat_value"], errors="coerce")
        m["stat_value"] = pd.to_numeric(m["stat_value"], errors="coerce")
        s_team = s.groupby("team_name", as_index=False)["stat_value"].sum()
        m_team = m.groupby("team_name", as_index=False)["stat_value"].sum()
        merged = s_team.merge(m_team, on="team_name", how="outer", suffixes=("_scored", "_missed")).fillna(0)
        merged[config["xlabel"]] = 100 * merged["stat_value_scored"] / (merged["stat_value_scored"] + merged["stat_value_missed"])
        merged[config["xlabel"]] = merged[config["xlabel"]].round(2)
        merged = merged.sort_values(config["xlabel"], ascending=False)
        return merged, config["xlabel"]
    elif t == "custom_shot_ratio":
        inside = df[df["stat_name"] == config["stats"][0]].copy()
        outside = df[df["stat_name"] == config["stats"][1]].copy()
        inside["stat_value"] = pd.to_numeric(inside["stat_value"], errors="coerce")
        outside["stat_value"] = pd.to_numeric(outside["stat_value"], errors="coerce")
        inside_team = inside.groupby("team_name", as_index=False)["stat_value"].sum()
        outside_team = outside.groupby("team_name", as_index=False)["stat_value"].sum()
        merged = inside_team.merge(outside_team, on="team_name", how="outer", suffixes=("_inside", "_outside")).fillna(0)
        merged[config["xlabel"]] = merged["stat_value_inside"] / merged["stat_value_outside"].replace(0, pd.NA)
        merged = merged.sort_values(config["xlabel"], ascending=False)
        return merged, config["xlabel"]
    elif t == "success_total_ratio":
        stat_df = df[df["stat_name"] == config["stats"][0]].copy()
        stat_df[["success", "total"]] = stat_df["stat_value"].astype(str).str.split("/", expand=True)
        stat_df["success"] = pd.to_numeric(stat_df["success"], errors="coerce")
        stat_df["total"] = pd.to_numeric(stat_df["total"], errors="coerce")
        team_grouped = stat_df.groupby("team_name", as_index=False).agg({"success": "sum", "total": "sum"})
        team_grouped[config["xlabel"]] = 100 * team_grouped["success"] / team_grouped["total"]
        team_grouped[config["xlabel"]] = team_grouped[config["xlabel"]].round(2)
        team_grouped = team_grouped.sort_values(config["xlabel"], ascending=False)
        return team_grouped, config["xlabel"]
    elif t == "foul_difference":
        fouls_df = master_df[master_df["stat_name"] == "Fouls"].copy()
        fouls_df["home_team_stat"] = pd.to_numeric(fouls_df["home_team_stat"], errors="coerce")
        fouls_df["away_team_stat"] = pd.to_numeric(fouls_df["away_team_stat"], errors="coerce")
        fouls_df = fouls_df.dropna(subset=["home_team_stat", "away_team_stat"])
        fouls_home = fouls_df.groupby("home_team").agg(
            Committed=("home_team_stat", "sum"),
            Suffered=("away_team_stat", "sum")
        ).reset_index().rename(columns={"home_team": "team"})
        fouls_away = fouls_df.groupby("away_team").agg(
            Committed=("away_team_stat", "sum"),
            Suffered=("home_team_stat", "sum")
        ).reset_index().rename(columns={"away_team": "team"})
        total_fouls_summary = (
            pd.concat([fouls_home, fouls_away], axis=0)
            .groupby("team", as_index=False)
            .sum()
        )
        total_fouls_summary[config["xlabel"]] = total_fouls_summary["Committed"] - total_fouls_summary["Suffered"]
        total_fouls_summary = total_fouls_summary.sort_values(config["xlabel"], ascending=False)
        total_fouls_summary = total_fouls_summary.rename(columns={"team": "team_name"})
        return total_fouls_summary, config["xlabel"]
    elif t == "cards_per_foul":
        yc_df = df[df["stat_name"] == config["stats"][0]].copy()
        rc_df = df[df["stat_name"] == config["stats"][1]].copy()
        fouls_df = df[df["stat_name"] == config["stats"][2]].copy()
        yc_df["stat_value"] = pd.to_numeric(yc_df["stat_value"], errors="coerce")
        rc_df["stat_value"] = pd.to_numeric(rc_df["stat_value"], errors="coerce")
        fouls_df["stat_value"] = pd.to_numeric(fouls_df["stat_value"], errors="coerce")
        cards_team = (
            yc_df.groupby("team_name", as_index=False)["stat_value"].sum()
            .rename(columns={"stat_value": "yellow"})
        ).merge(
            rc_df.groupby("team_name", as_index=False)["stat_value"].sum()
            .rename(columns={"stat_value": "red"}),
            on="team_name", how="outer"
        ).fillna(0)
        fouls_team = fouls_df.groupby("team_name", as_index=False)["stat_value"].sum().rename(columns={"stat_value": "fouls"})
        merged = pd.merge(cards_team, fouls_team, on="team_name", how="inner")
        merged[config["xlabel"]] = (merged["yellow"] + merged["red"]) / merged["fouls"].replace(0, pd.NA)
        merged[config["xlabel"]] = merged[config["xlabel"]].round(3)
        merged = merged.sort_values(config["xlabel"], ascending=False)
        return merged, config["xlabel"]
    else:
        raise NotImplementedError(f"Unknown stat type: {t}")

def _plot_stat(team_df, value_col, config, highlight_team, season, league, max_week):
    norm = plt.Normalize(team_df[value_col].min(), team_df[value_col].max())
    cmap = plt.get_cmap("coolwarm_r")
    colors = cmap(norm(team_df[value_col]))

    fig, ax = plt.subplots(figsize=(7, len(team_df) * 0.5))
    ax.barh(team_df["team_name"], team_df[value_col], color=colors, align="center")
    ax.set_xlabel(config["xlabel"], labelpad=20)
    ax.set_title(
        f"{season} {league}\n{config['title']} for {highlight_team}\n(up to Week {max_week})",
        fontsize=10, fontweight="bold"
    )
    if "ylim" in config:
        ax.set_xlim(config["ylim"])
    ax.invert_yaxis()
    plt.tight_layout()
    for label in ax.get_yticklabels():
        if label.get_text() == highlight_team:
            label.set_fontweight("bold")
            label.set_color("black")
        else:
            label.set_fontweight("normal")
            label.set_color("gray")
    ax.grid(True, linestyle="--", alpha=0.7)
    st.pyplot(fig)

def run(team: str, country: str, league: str, season: str):
    match_df, match_stats_df = require_session_data("match_data", "match_stats_data")

    match_df = filter_matches_by_status(match_df, "Ended")

    max_week = match_df["week"].max()

    match_df = match_df[["game_id", "home_team", "away_team"]]
    match_stats_df = match_stats_df[match_stats_df["period"] == "ALL"]

    percent_keywords = ["Ball possession", "Tackles won", "Duels"]
    parenthesis_keywords = ["Final third phase", "Long balls", "Crosses", "Ground duels", "Aerial duels", "Dribbles"]
    target_columns = ["home_team_stat", "away_team_stat"]

    match_stats_df = _clean_percent_columns(match_stats_df, percent_keywords, target_columns)
    match_stats_df = _clean_parenthesis_columns(match_stats_df, parenthesis_keywords, target_columns)

    master_df = match_stats_df.merge(match_df, on="game_id")

    all_teams = set(master_df["home_team"]).union(set(master_df["away_team"]))
    if team not in all_teams:
        st.warning(f"No data available yet for {team} in {season} {league}.")
        return

    all_stats_df_list = []
    for stat in master_df["stat_name"].unique():
        stat_df = master_df[master_df["stat_name"] == stat]
        temp_df = pd.DataFrame({
            "team_name": pd.concat([stat_df["home_team"], stat_df["away_team"]]),
            "stat_name": [stat] * len(stat_df) * 2,
            "stat_value": pd.concat([stat_df["home_team_stat"], stat_df["away_team_stat"]])
        })
        all_stats_df_list.append(temp_df)
    result_all_stats_df = pd.concat(all_stats_df_list, ignore_index=True).reset_index(drop=True)

    selected_stat = st.selectbox(
        "Statistic", list(STAT_CONFIG.keys()), index=None, placeholder="Please select a statistic"
    )
    if selected_stat:
        config = STAT_CONFIG[selected_stat]
        if config["type"] == "foul_difference":
            team_df, value_col = _compute_stat(result_all_stats_df, config, master_df)
        else:
            team_df, value_col = _compute_stat(result_all_stats_df, config)
        _plot_stat(team_df, value_col, config, team, season, league, max_week)