import streamlit as st
import pandas as pd
from modules.homepage import get_data
from code.utils.plotters import plot_boxplot, plot_stacked_bar_chart, plot_stacked_horizontal_bar, plot_horizontal_bar

def clean_percent_columns(dataframe, columns_to_check, target_columns):
    for index, row in dataframe.iterrows():
        if any(keyword in row["stat_name"] for keyword in columns_to_check):
            for col in target_columns:
                dataframe.at[index, col] = row[col].replace("%", "").strip()
    return dataframe

def clean_parenthesis_columns(dataframe, columns_to_check, target_columns):
    for index, row in dataframe.iterrows():
        if any(keyword in row["stat_name"] for keyword in columns_to_check):
            for col in target_columns:
                if "(" in row[col]:
                    dataframe.at[index, col] = row[col].split("(")[0].strip()
    return dataframe

def create_performance_plot(master_df, result_all_stats_df, subcategory, last_round):
    if subcategory == "Ball Possession":
        possession_data = result_all_stats_df[result_all_stats_df["stat_name"] == "Ball possession"]

        median_possession_by_team = possession_data.groupby("team_name")["stat_value"].median().reset_index()
        median_possession_by_team = median_possession_by_team.sort_values("stat_value", ascending=False)

        sorted_team_names = median_possession_by_team["team_name"]

        plot_boxplot(
            data=possession_data,
            x="stat_value",
            y="team_name",
            title=f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Ball Possession by Team",
            xlabel="Ball Possession (%) (Median)",
            ylabel="",
            ordered_labels=sorted_team_names
        )
    elif subcategory == "Pass Success Rate":
        passing_data = result_all_stats_df[result_all_stats_df["stat_name"].isin(["Passes", "Accurate passes"])]

        team_passing_stats = passing_data.pivot_table(
            index="team_name",
            columns="stat_name",
            values="stat_value",
            aggfunc="sum",
            fill_value=0
        )

        team_passing_stats["Total passes"] = team_passing_stats["Passes"]
        team_passing_stats["Inaccurate passes"] = team_passing_stats["Total passes"] - team_passing_stats["Accurate passes"]

        plot_stacked_bar_chart(
            data=team_passing_stats,
            stat_columns=["Accurate passes", "Inaccurate passes"],
            total_column="Total passes",
            title=f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Passing Accuracy by Team",
            xlabel="Number of Passes",
            ylabel="",
            colors={"Accurate passes": "#4169E1", "Inaccurate passes": "#CD5C5C"},
            sort_by="Total passes",
            ascending=True
        )
    elif subcategory == "Big Chances Scored/Missed":
        big_chances_data = result_all_stats_df[
            result_all_stats_df["stat_name"].isin(["Big chances scored", "Big chances missed"])
        ]

        team_big_chances_stats = big_chances_data.pivot_table(
            index="team_name",
            columns="stat_name",
            values="stat_value",
            aggfunc="sum"
        ).fillna(0)

        team_big_chances_stats["Total big chances"] = team_big_chances_stats.sum(axis=1)

        team_big_chances_stats = team_big_chances_stats.sort_values("Total big chances")

        plot_stacked_horizontal_bar(
            data=team_big_chances_stats,
            stat_columns=["Big chances scored", "Big chances missed"],
            total_column="Total big chances",
            title=f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Big Chance Conversion Rate by Team",
            xlabel="Number of Big Chances",
            ylabel="",
            colors={
                "Big chances scored": "#4169E1",
                "Big chances missed": "#CD5C5C"
            }
        )
    elif subcategory == "Shot Accuracy":
        shooting_data = result_all_stats_df[
            result_all_stats_df["stat_name"].isin(
                ["Shots on target", "Shots off target", "Blocked shots", "Hit woodwork"]
            )
        ]

        team_shooting_stats = shooting_data.pivot_table(
            index="team_name",
            columns="stat_name",
            values="stat_value",
            aggfunc="sum",
            fill_value=0
        )

        team_shooting_stats["Total shots"] = team_shooting_stats.sum(axis=1)

        shooting_categories = ["Shots on target", "Shots off target", "Blocked shots", "Hit woodwork"]

        for category in shooting_categories:
            team_shooting_stats[f"{category} (%)"] = (team_shooting_stats[category] / team_shooting_stats["Total shots"]) * 100

        team_shooting_stats = team_shooting_stats.sort_values("Total shots")

        plot_stacked_horizontal_bar(
            data=team_shooting_stats,
            stat_columns=shooting_categories,
            total_column="Total shots",
            title=f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Shooting Accuracy by Team",
            xlabel="Number of Shots",
            ylabel="",
            colors={
                "Shots on target": "#4169E1",
                "Shots off target": "#CD5C5C",
                "Blocked shots": "#FFA07A",
                "Hit woodwork": "#FFD700"
            }
        )
    elif subcategory == "Shot Ratio Inside/Outside Box":
        penalty_area_shooting_data = result_all_stats_df[
            result_all_stats_df["stat_name"].isin(["Shots inside box", "Shots outside box"])
        ]

        team_penalty_area_stats = penalty_area_shooting_data.pivot_table(
            index="team_name",
            columns="stat_name",
            values="stat_value",
            aggfunc="sum",
            fill_value=0
        )

        team_penalty_area_stats["Total shots"] = team_penalty_area_stats.sum(axis=1)

        shooting_categories = ["Shots inside box", "Shots outside box"]

        for category in shooting_categories:
            team_penalty_area_stats[f"{category} (%)"] = (team_penalty_area_stats[category] / team_penalty_area_stats["Total shots"]) * 100

        team_penalty_area_stats = team_penalty_area_stats.sort_values("Total shots")

        plot_stacked_horizontal_bar(
            data=team_penalty_area_stats,
            stat_columns=shooting_categories,
            total_column="Total shots",
            title=f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Shot Preference by Penalty Area by Team",
            xlabel="Number of Shots",
            ylabel="",
            colors={
                "Shots inside box": "#4169E1",
                "Shots outside box": "#CD5C5C"
            }
        )
    elif subcategory == "Touches in Penalty Area":
        penalty_area_touch_data = result_all_stats_df[
            result_all_stats_df["stat_name"] == "Touches in penalty area"
        ].groupby("team_name", as_index=False)["stat_value"].sum()

        penalty_area_touch_data["stat_value"] = penalty_area_touch_data["stat_value"].astype(int)

        plot_horizontal_bar(
            data=penalty_area_touch_data,
            x="stat_value",
            y="team_name",
            title=f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Touches in the Penalty Area by Team",
            xlabel="Touches in the Penalty Area",
            ylabel="",
            sort_by="stat_value",
            ascending=True,
            calculate_percentages=False,
            cmap_name="coolwarm_r",
            is_int=True
        )
    elif subcategory == "Final Third Entries":
        final_third_entry_data = result_all_stats_df[
            result_all_stats_df["stat_name"] == "Final third entries"
        ].groupby("team_name", as_index=False)["stat_value"].sum()

        final_third_entry_data["stat_value"] = final_third_entry_data["stat_value"].astype(int)

        plot_horizontal_bar(
            data=final_third_entry_data,
            x="stat_value",
            y="team_name",
            title=f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Entries into the Final Third by Team",
            xlabel="Entries into the Final Third",
            ylabel="",
            sort_by="stat_value",
            ascending=True,
            calculate_percentages=False,
            cmap_name="coolwarm_r",
            is_int=True
        )
    elif subcategory == "Successful Actions in Final Third":
        final_third_action_data = result_all_stats_df[
            result_all_stats_df["stat_name"] == "Final third phase"
        ]

        final_third_action_data[["Successful Actions", "Total Actions"]] = (
            final_third_action_data["stat_value"]
            .str.split("/", expand=True)
            .astype(int)
        )

        final_third_action_data["Unsuccessful Actions"] = (
            final_third_action_data["Total Actions"] - final_third_action_data["Successful Actions"]
        )

        team_final_third_actions = final_third_action_data.groupby("team_name", as_index=True).sum()

        plot_stacked_horizontal_bar(
            data=team_final_third_actions,
            stat_columns=["Successful Actions", "Unsuccessful Actions"],
            total_column="Total Actions",
            title=f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Final Third Action Efficiency by Team",
            xlabel="Number of Actions in the Final Third",
            ylabel="",
            colors={
                "Successful Actions": "#4169E1",
                "Unsuccessful Actions": "#CD5C5C"
            },
            sort_by="Total Actions",
            ascending=True
        )
    elif subcategory == "Difference Between Fouls Committed and Suffered":
        foul_data = master_df[master_df["stat_name"] == "Fouls"].copy()

        foul_data["home_team_stat"] = pd.to_numeric(foul_data["home_team_stat"], errors="coerce")
        foul_data["away_team_stat"] = pd.to_numeric(foul_data["away_team_stat"], errors="coerce")

        foul_data = foul_data.dropna(subset=["home_team_stat", "away_team_stat"])

        fouls_home = foul_data.groupby("home_team").agg(
            Committed=("home_team_stat", "sum"),
            Suffered=("away_team_stat", "sum")
        ).reset_index().rename(columns={"home_team": "team"})

        fouls_away = foul_data.groupby("away_team").agg(
            Committed=("away_team_stat", "sum"),
            Suffered=("home_team_stat", "sum")
        ).reset_index().rename(columns={"away_team": "team"})

        total_fouls_summary = (
            pd.concat([fouls_home, fouls_away], axis=0)
            .groupby("team", as_index=False)
            .sum()
        )

        total_fouls_summary["Foul Difference"] = (
            total_fouls_summary["Committed"] - total_fouls_summary["Suffered"]
        )
        total_fouls_summary = total_fouls_summary.sort_values("Foul Difference", ascending=True)

        plot_horizontal_bar(
            data=total_fouls_summary,
            x="Foul Difference",
            y="team",
            title=f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Difference Between Fouls Committed and Fouls Suffered by Team",
            xlabel="Foul Difference",
            ylabel="",
            calculate_percentages=False,
            sort_by="Foul Difference",
            ascending=True,
            cmap_name="coolwarm",
            is_int=True
        )
    elif subcategory == "Cards Per Foul":
        fouls_data = result_all_stats_df[
            result_all_stats_df["stat_name"] == "Fouls"
        ].groupby("team_name", as_index=False)["stat_value"].sum().rename(columns={"stat_value": "Number of Fouls"})

        yellow_cards_data = result_all_stats_df[
            result_all_stats_df["stat_name"] == "Yellow cards"
        ].groupby("team_name", as_index=False)["stat_value"].sum().rename(columns={"stat_value": "Number of Yellow Cards"})

        red_cards_data = result_all_stats_df[
            result_all_stats_df["stat_name"] == "Red cards"
        ].groupby("team_name", as_index=False)["stat_value"].sum().rename(columns={"stat_value": "Number of Red Cards"})

        cards_data = pd.merge(
            yellow_cards_data,
            red_cards_data,
            on="team_name",
            how="outer"
        ).fillna(0)

        cards_data["Total Number of Cards"] = cards_data["Number of Yellow Cards"] + cards_data["Number of Red Cards"]

        merged_data = pd.merge(
            fouls_data, cards_data[["team_name", "Total Number of Cards"]],
            on="team_name"
        )

        merged_data["Cards per Foul"] = merged_data["Total Number of Cards"] / merged_data["Number of Fouls"]
        merged_data["Cards per Foul"] = pd.to_numeric(merged_data["Cards per Foul"], errors="coerce")

        plot_horizontal_bar(
            data=merged_data,
            x="Cards per Foul",
            y="team_name",
            title=f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Number of Cards (Yellow and Red) per Foul Committed by Team",
            xlabel="Cards per Foul (Yellow and Red)",
            ylabel="",
            calculate_percentages=False,
            sort_by="Cards per Foul",
            ascending=True,
            cmap_name="coolwarm",
            is_int=False
        )
    elif subcategory == "Accurate Long Pass Rate":
        long_pass_data = result_all_stats_df[result_all_stats_df["stat_name"] == "Long balls"]

        long_pass_data[["Successful Long Pass", "Total Long Pass"]] = (
            long_pass_data["stat_value"]
            .str.split("/", expand=True)
            .astype(int)
        )

        long_pass_data["Unsuccessful Long Pass"] = (
            long_pass_data["Total Long Pass"] - long_pass_data["Successful Long Pass"]
        )

        grouped_long_pass_data = long_pass_data.groupby("team_name", as_index=True).sum()

        plot_stacked_horizontal_bar(
            data=grouped_long_pass_data,
            stat_columns=["Successful Long Pass", "Unsuccessful Long Pass"],
            total_column="Total Long Pass",
            title=f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Successful and Unsuccessful Long Pass Rate by Team",
            xlabel="Number of Long Passes",
            ylabel="",
            colors={
                "Successful Long Pass": "#4169E1",
                "Unsuccessful Long Pass": "#CD5C5C"
            },
            sort_by="Total Long Pass",
            ascending=True
        )
    elif subcategory == "Accurate Cross Rate":
        crossing_data = result_all_stats_df[result_all_stats_df["stat_name"] == "Crosses"]

        crossing_data[["Successful Crosses", "Total Crosses"]] = (
            crossing_data["stat_value"]
            .str.split("/", expand=True)
            .astype(int)
        )

        crossing_data["Unsuccessful Crosses"] = (
            crossing_data["Total Crosses"] - crossing_data["Successful Crosses"]
        )

        grouped_crossing_data = crossing_data.groupby("team_name", as_index=True).sum()

        plot_stacked_horizontal_bar(
            data=grouped_crossing_data,
            stat_columns=["Successful Crosses", "Unsuccessful Crosses"],
            total_column="Total Crosses",
            title=f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season – Successful and Unsuccessful Cross Rate by Team",
            xlabel="Number of Crosses",
            ylabel="",
            colors={
                "Successful Crosses": "#4169E1",
                "Unsuccessful Crosses": "#CD5C5C"
            },
            sort_by="Total Crosses",
            ascending=True
        )

def main(subcategory):
    try:

        match_data_df = get_data("match_data")
        match_stats_data_df = get_data("match_stats_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]
        match_data_df = match_data_df[["game_id","home_team","away_team"]]

        match_stats_data_df = match_stats_data_df[match_stats_data_df["period"] == "ALL"]

        percent_keywords = ["Ball possession", "Tackles won", "Duels"]
        parenthesis_keywords = ["Final third phase", "Long balls", "Crosses", "Ground duels", "Aerial duels", "Dribbles"]
        target_columns = ["home_team_stat", "away_team_stat"]

        match_stats_data_df = clean_percent_columns(match_stats_data_df, percent_keywords, target_columns)
        match_stats_data_df = clean_parenthesis_columns(match_stats_data_df, parenthesis_keywords, target_columns)

        master_df = match_stats_data_df.merge(
            match_data_df,
            on="game_id"
        )

        all_stats_df_list = []

        for stat in master_df["stat_name"].unique():
            stat_df = master_df[master_df["stat_name"] == stat]
            temp_df = pd.DataFrame({
                "team_name": pd.concat([stat_df["home_team"], stat_df["away_team"]]),
                "stat_name": [stat] * len(stat_df) * 2,
                "stat_value": pd.concat([stat_df["home_team_stat"], stat_df["away_team_stat"]])
            })
            all_stats_df_list.append(temp_df)

        result_all_stats_df = pd.concat(all_stats_df_list, ignore_index=True)
        result_all_stats_df = result_all_stats_df.reset_index(drop=True)

        result_all_stats_df.loc[~result_all_stats_df["stat_value"].str.contains("/", na=False), "stat_value"] = \
            pd.to_numeric(result_all_stats_df["stat_value"], errors="coerce")

        last_round = master_df["week"].max()

        create_performance_plot(master_df, result_all_stats_df, subcategory, last_round)

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