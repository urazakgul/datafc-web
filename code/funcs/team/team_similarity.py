import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from modules.homepage import get_data
from code.utils.helpers import add_footer
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

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

def process_exceptions(dataframe, exc_list):
    exception_handling = {
        "Ball Possession": "average",
        "Tackles won": "average",
        "Duels": "average",
        "Final third phase": "split_and_average",
        "Long balls": "split_and_average",
        "Crosses": "split_and_average",
        "Ground duels": "split_and_average",
        "Aerial duels": "split_and_average",
        "Dribbles": "split_and_average",
    }

    processed_df_list = []

    for stat_name, handling_type in exception_handling.items():
        stat_df = dataframe[dataframe["stat_name"] == stat_name].copy()

        if handling_type == "average":
            stat_df = stat_df.groupby("team_name", as_index=False).agg({"stat_value": "mean"})
            stat_df["stat_name"] = stat_name
            processed_df_list.append(stat_df)

        elif handling_type == "split_and_average":
            success_rate = []
            total_values = []

            for value in stat_df["stat_value"]:
                if isinstance(value, str) and "/" in value:
                    try:
                        success, total = map(int, value.split("/"))
                        success_rate.append(success / total * 100)
                        total_values.append(total)
                    except ValueError:
                        success_rate.append(None)
                        total_values.append(None)
                else:
                    success_rate.append(None)
                    total_values.append(None)

            stat_df["stat_value_success"] = success_rate
            stat_df["stat_value_total"] = total_values

            success_avg_df = stat_df.groupby("team_name", as_index=False).agg({"stat_value_success": "mean"})
            success_avg_df["stat_name"] = f"{stat_name}-Success"

            total_sum_df = stat_df.groupby("team_name", as_index=False).agg({"stat_value_total": "sum"})
            total_sum_df["stat_name"] = f"{stat_name}-Total"

            processed_df_list.append(success_avg_df.rename(columns={"stat_value_success": "stat_value"}))
            processed_df_list.append(total_sum_df.rename(columns={"stat_value_total": "stat_value"}))

    return pd.concat(processed_df_list, ignore_index=True)

def update_game_stats_categories(match_stats_data_df):
    CATEGORY_MAPPINGS = {
        "Ball possession": "Passes",
        "Expected goals": "Attack",
        "Big chances": "Attack",
        "Goalkeeper saves": "Goalkeeping",
        "Corner kicks": "Attack",
        "Fouls": "Defending",
        "Passes": "Passes",
        "Free kicks": "Attack",
        "Yellow cards": "Defending",
        "Red cards": "Defending"
    }

    REMOVE_STATS = ["Total shots", "Tackles"]

    match_stats_data_df.loc[
        (match_stats_data_df["group_name"] == "Match overview") &
        (match_stats_data_df["stat_name"].isin(CATEGORY_MAPPINGS.keys())),
        "group_name"
    ] = match_stats_data_df["stat_name"].map(CATEGORY_MAPPINGS)

    match_stats_data_df = match_stats_data_df[
        ~(
            (match_stats_data_df["group_name"] == "Match overview") &
            (match_stats_data_df["stat_name"].isin(REMOVE_STATS))
        )
    ]

    return match_stats_data_df

def create_team_similarity_plot(similarity_df, team, last_round, selected_categories, similarity_algorithm, top_features_pc1, top_features_pc2):
    if similarity_algorithm == "Cosine Similarity":
        norm = plt.Normalize(similarity_df["similarity"].min(), similarity_df["similarity"].max())
        colors = cm.coolwarm_r(norm(similarity_df["similarity"]))

        fig, ax = plt.subplots(figsize=(12, 12))
        bars = ax.barh(similarity_df["team_name"], similarity_df["similarity"], color=colors)

        ax.set_xlabel("Similarity Score", fontsize=12, labelpad=15)
        ax.set_ylabel("")
        title = f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Team Similarity Ranking to {team}"
        subtitle = f"Based on {similarity_algorithm} Algorithm and {', '.join(selected_categories)} Categories"
        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold",
            pad=40
        )
        ax.text(
            0.5, 1.02,
            subtitle,
            fontsize=8,
            fontstyle="italic",
            color="gray",
            ha="center",
            va="bottom",
            transform=ax.transAxes
        )

        add_footer(fig, y=-0.03)

        ax.axvline(x=0, color="black", linewidth=1, alpha=0.3)

        for similarity, bar in zip(similarity_df["similarity"], bars):
            x_position = bar.get_width() + 0.01 if similarity >= 0 else bar.get_width() - 0.01
            ha_value = "left" if similarity >= 0 else "right"

            ax.text(
                x_position,
                bar.get_y() + bar.get_height() / 2,
                f"{similarity:.2f}",
                va="center",
                ha=ha_value,
                fontsize=10
            )

        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        st.pyplot(fig)

    elif similarity_algorithm == "Principal Component Analysis":
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(
            similarity_df["PC1"], similarity_df["PC2"],
            c="blue", alpha=0.6, edgecolor="k"
        )

        ax.set_xlabel("PC1", fontsize=12, labelpad=15)
        ax.set_ylabel("PC2", fontsize=12, labelpad=15)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)

        title = f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - PCA-Based Team Positioning"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=50)

        best_loadings_txt = f"Top 5 Loadings for PC1: {', '.join(top_features_pc1)}\nTop 5 Loadings for PC2: {', '.join(top_features_pc2)}"
        ax.text(
            0.5, 1.04,
            best_loadings_txt,
            fontsize=8,
            fontstyle="italic",
            color="gray",
            ha="center",
            va="bottom",
            transform=ax.transAxes
        )

        def getImage(path):
            return OffsetImage(plt.imread(path), zoom=.3, alpha=1)

        for index, row in similarity_df.iterrows():
            logo_path = f"./imgs/{st.session_state['selected_league']}_logos/{index}.png"
            ab = AnnotationBbox(getImage(logo_path), (row["PC1"], row["PC2"]), frameon=False)
            ax.add_artist(ab)

        add_footer(fig, y=-0.03)
        plt.tight_layout()
        ax.grid(True, linestyle="--", alpha=0.7)

        st.pyplot(fig)

def main(team, selected_categories, similarity_algorithm):

    try:

        match_data_df = get_data("match_data")
        match_stats_data_df = get_data("match_stats_data")

        match_data_df = match_data_df[match_data_df["status"].isin(["Ended"])]

        match_stats_data_df = update_game_stats_categories(match_stats_data_df)

        match_data_df = match_data_df[["game_id", "home_team", "away_team"]]

        match_stats_data_df = match_stats_data_df[match_stats_data_df["period"] == "ALL"]
        match_stats_data_df = match_stats_data_df.rename(
            columns={"home_team": "home_team_stat", "away_team": "away_team_stat"}
        )

        percent_keywords = ["Ball possession", "Tackles won", "Duels"]
        parenthesis_keywords = ["Final third phase", "Long balls", "Crosses", "Ground duels", "Aerial duels", "Dribbles"]
        target_columns = ["home_team_stat", "away_team_stat"]

        exc_list = [
            "Ball Possession",
            "Tackles won",
            "Duels",
            "Final third phase",
            "Long balls",
            "Crosses",
            "Ground duels",
            "Aerial duels",
            "Dribbles"
        ]

        match_stats_data_df = clean_percent_columns(match_stats_data_df, percent_keywords, target_columns)
        match_stats_data_df = clean_parenthesis_columns(match_stats_data_df, parenthesis_keywords, target_columns)

        match_stats_data_df.loc[
            ~match_stats_data_df["home_team_stat"].str.contains("/", na=False), "home_team_stat"
        ] = pd.to_numeric(match_stats_data_df["home_team_stat"], errors="coerce")

        match_stats_data_df.loc[
            ~match_stats_data_df["away_team_stat"].str.contains("/", na=False), "away_team_stat"
        ] = pd.to_numeric(match_stats_data_df["away_team_stat"], errors="coerce")

        master_df = match_stats_data_df.merge(match_data_df, on="game_id")

        all_stats_df_list = []

        for stat in master_df["stat_name"].unique():
            stat_df = master_df[master_df["stat_name"] == stat]
            temp_df = pd.DataFrame(
                {
                    "team_name": pd.concat([stat_df["home_team"], stat_df["away_team"]]),
                    "stat_name": [stat] * len(stat_df) * 2,
                    "group_name": pd.concat([stat_df["group_name"], stat_df["group_name"]]),
                    "stat_value": pd.concat([stat_df["home_team_stat"], stat_df["away_team_stat"]]),
                }
            )
            all_stats_df_list.append(temp_df)

        result_all_stats_df = pd.concat(all_stats_df_list, ignore_index=True)
        result_all_stats_df = result_all_stats_df.reset_index(drop=True)

        filtered_result_selected_stats_df = result_all_stats_df[
            (result_all_stats_df["group_name"].isin(selected_categories))
        ]

        exc_processed_df = process_exceptions(filtered_result_selected_stats_df, exc_list)

        remaining_stats_df = filtered_result_selected_stats_df[
            ~filtered_result_selected_stats_df["stat_name"].isin(exc_list)
        ]

        aggregated_df = remaining_stats_df.groupby(["team_name", "stat_name"], as_index=False)["stat_value"].sum()

        final_df = pd.concat([exc_processed_df, aggregated_df], ignore_index=True)

        pivot_df = final_df.pivot(index="team_name", columns="stat_name", values="stat_value").fillna(0)

        if similarity_algorithm == "Cosine Similarity":
            scaler = StandardScaler()
            normalized_pivot_df = pd.DataFrame(
                scaler.fit_transform(pivot_df), index=pivot_df.index, columns=pivot_df.columns
            )

            target_vector = normalized_pivot_df.loc[team].values.reshape(1, -1)
            similarity_scores = cosine_similarity(target_vector, normalized_pivot_df)[0]
            similarity_df = pd.DataFrame(
                {"team_name": normalized_pivot_df.index, "similarity": similarity_scores}
            ).sort_values(by="similarity", ascending=False)
            similarity_df = similarity_df[similarity_df["team_name"] != team]

        elif similarity_algorithm == "Principal Component Analysis":
            scaler = StandardScaler()
            normalized_pivot_df = pd.DataFrame(
                scaler.fit_transform(pivot_df), index=pivot_df.index, columns=pivot_df.columns
            )

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(normalized_pivot_df)
            similarity_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=normalized_pivot_df.index)

            loadings = pd.DataFrame(
                pca.components_.T,
                columns=["PC1", "PC2"],
                index=pivot_df.columns
            )

            top_loadings = loadings.abs().nlargest(5, ["PC1", "PC2"])
            top_features_pc1 = top_loadings["PC1"].nlargest(5).index.tolist()
            top_features_pc2 = top_loadings["PC2"].nlargest(5).index.tolist()

        last_round = master_df["week"].max()

        create_team_similarity_plot(
            similarity_df,
            team,
            last_round,
            selected_categories,
            similarity_algorithm,
            top_features_pc1 if similarity_algorithm=="Principal Component Analysis" else None,
            top_features_pc2 if similarity_algorithm=="Principal Component Analysis" else None
        )

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