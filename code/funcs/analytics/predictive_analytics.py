import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from modules.homepage import get_data
from code.utils.helpers import add_footer
from code.models.dixon_coles import solve_parameters_cached, dixon_coles_simulate_match_cached
from code.models.bradley_terry import solve_bt_ratings_cached, bt_forecast_match_cached
from config import PLOT_STYLE

plt.style.use(PLOT_STYLE)

def create_predictive_analytics_plot(
        model_df,
        selected_model,
        home_team,
        away_team,
        selected_week,
        plot_type,
        first_n_goals,
        bt_prob,
        params=None
):

    if selected_model == "Dixon-Coles":
        home_win_prob = np.sum(np.tril(model_df, -1)) * 100
        draw_prob = np.sum(np.diag(model_df)) * 100
        away_win_prob = np.sum(np.triu(model_df, 1)) * 100

        percentage_matrix = model_df * 100

        if plot_type == "Matrix":
            fig, ax = plt.subplots(figsize=(10, 8))

            sns.heatmap(
                percentage_matrix,
                annot=True,
                fmt=".1f",
                cmap="Reds",
                cbar=False,
                ax=ax
            )

            ax.set_xlabel(f"{away_team} (Away) Goal Count", labelpad=20, fontsize=12)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()

            ax.set_ylabel(f"{home_team} (Home) Goal Count", fontsize=12)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            ax.set_title(
                f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Match Outcome Probabilities for Week {selected_week}",
                fontsize=16,
                fontweight="bold",
                pad=45
            )

            plt.suptitle(
                f"{home_team} Win: %{home_win_prob:.1f} | Draw: %{draw_prob:.1f} | {away_team} Win: %{away_win_prob:.1f}",
                fontsize=10,
                y=0.89
            )

            add_footer(fig, x=0.98, y=-0.05, fontsize=8, extra_text=f"Results are based on the {selected_model} model.")
            plt.tight_layout()

        elif plot_type == "Ranked":
            bar_data = []
            for i in range(percentage_matrix.shape[0]):
                for j in range(percentage_matrix.shape[1]):
                    bar_data.append({
                        "Home Goals": int(i),
                        "Away Goals": int(j),
                        "Probability": percentage_matrix[i, j]
                    })

            bar_data = pd.DataFrame(bar_data)
            bar_data = bar_data.sort_values("Probability", ascending=False).head(20)

            norm = mcolors.Normalize(vmin=bar_data["Probability"].min(), vmax=bar_data["Probability"].max())
            cmap = plt.get_cmap("Reds")
            colors = [cmap(norm(prob)) for prob in bar_data["Probability"]]

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.barh(
                bar_data.apply(lambda x: f"{int(x['Home Goals'])} - {int(x['Away Goals'])}", axis=1),
                bar_data["Probability"],
                color=colors,
                edgecolor="black"
            )

            ax.set_title(
                f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Top 20 Predicted Scorelines for Week {selected_week}",
                fontsize=16,
                fontweight="bold",
                pad=50
            )
            plt.suptitle(
                f"{home_team} vs. {away_team}",
                fontsize=16,
                fontweight="bold",
                y=0.91
            )
            ax.set_xlabel("Probability (%)", labelpad=20, fontsize=12)
            ax.set_ylabel("Goal Combinations", labelpad=20, fontsize=12)
            ax.invert_yaxis()

            ax.grid(True, linestyle="--", alpha=0.7)

            add_footer(fig, x=0.98, y=-0.02, fontsize=8, extra_text=f"Results are based on the {selected_model} model.")
            plt.tight_layout()

        elif plot_type == "Summary":
            bar_data = []
            for i in range(percentage_matrix.shape[0]):
                for j in range(percentage_matrix.shape[1]):
                    bar_data.append({
                        "Home Goals": int(i),
                        "Away Goals": int(j),
                        "Probability": percentage_matrix[i, j]
                    })

            bar_data = pd.DataFrame(bar_data)
            top_combinations = bar_data.sort_values("Probability", ascending=False).head(first_n_goals)

            summary = {
                "Home Leads": sum(top_combinations["Home Goals"] > top_combinations["Away Goals"]),
                "Away Leads": sum(top_combinations["Home Goals"] < top_combinations["Away Goals"]),
                "Draw": sum(top_combinations["Home Goals"] == top_combinations["Away Goals"]),
                "Average Home Goals": top_combinations["Home Goals"].mean(),
                "Average Away Goals": top_combinations["Away Goals"].mean(),
                "Both Teams Scored": sum((top_combinations["Home Goals"] > 0) & (top_combinations["Away Goals"] > 0)),
                "No Goals": sum((top_combinations["Home Goals"] == 0) | (top_combinations["Away Goals"] == 0)),
                "Over 0.5": sum((top_combinations["Home Goals"] + top_combinations["Away Goals"]) > 0.5),
                "Over 1.5": sum((top_combinations["Home Goals"] + top_combinations["Away Goals"]) > 1.5),
                "Over 2.5": sum((top_combinations["Home Goals"] + top_combinations["Away Goals"]) > 2.5),
                "Over 3.5": sum((top_combinations["Home Goals"] + top_combinations["Away Goals"]) > 3.5),
                "Over 4.5": sum((top_combinations["Home Goals"] + top_combinations["Away Goals"]) > 4.5),
                "Under 0.5": sum((top_combinations["Home Goals"] + top_combinations["Away Goals"]) <= 0.5),
                "Under 1.5": sum((top_combinations["Home Goals"] + top_combinations["Away Goals"]) <= 1.5),
                "Under 2.5": sum((top_combinations["Home Goals"] + top_combinations["Away Goals"]) <= 2.5),
                "Under 3.5": sum((top_combinations["Home Goals"] + top_combinations["Away Goals"]) <= 3.5),
                "Under 4.5": sum((top_combinations["Home Goals"] + top_combinations["Away Goals"]) <= 4.5),
            }

            group_1 = {k: summary[k] for k in ["Home Leads", "Draw", "Away Leads"]}
            group_2 = {k: summary[k] for k in ["Average Home Goals", "Average Away Goals"]}
            group_3 = {k: summary[k] for k in ["Both Teams Scored", "No Goals"]}
            group_4 = {k: summary[k] for k in ["Over 0.5", "Over 1.5", "Over 2.5", "Over 3.5", "Over 4.5"]}
            group_5 = {k: summary[k] for k in ["Under 0.5", "Under 1.5", "Under 2.5", "Under 3.5", "Under 4.5"]}

            groups = [group_1, group_2, group_3, group_4, group_5]
            group_titles = ["Result", "Average Goals", "Both Teams to Score", "Over Betting", "Under Betting"]

            fig, axes = plt.subplots(5, 1, figsize=(12, 20))

            fig.suptitle(
                f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Summary of the Top {first_n_goals} Goal Combinations for Week {selected_week}",
                fontsize=16,
                fontweight="bold",
                y=1.04
            )
            fig.text(
                0.5, 1.00,
                f"{home_team} - {away_team}",
                ha="center",
                fontsize=14,
                fontweight="bold"
            )

            for ax, group, title in zip(axes, groups, group_titles):
                ax.bar(group.keys(), group.values(), color="indianred", edgecolor="black")
                ax.set_title(title, fontsize=14, fontweight="bold")
                ax.set_ylabel("")
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                ax.set_xticklabels(group.keys(), rotation=0, ha="center")

            add_footer(fig, x=0.98, y=-0.02, fontsize=8, extra_text=f"Results are based on the {selected_model} model.")
            plt.tight_layout()

        elif plot_type == "Team Strength":

            teams = list(set(
                k.split("_")[1] for k in params.keys() if "_" in k and (k.startswith("attack_") or k.startswith("defence_"))
            ))
            attack_strength = np.array([params.get(f"attack_{team}", 0) for team in teams])
            defense_strength = np.array([params.get(f"defence_{team}", 0) for team in teams])

            fig, ax = plt.subplots(figsize=(12, 8))

            mean_attack = np.mean(attack_strength)
            mean_defense = np.mean(defense_strength)

            ax.axvline(mean_attack, color="red", linestyle="dashed", linewidth=1, label="Average Attack Strength")
            ax.axhline(mean_defense, color="blue", linestyle="dashed", linewidth=1, label="Average Defense Strength")

            ax.scatter(attack_strength, defense_strength, color="red", s=100, edgecolor="black", alpha=0.7)

            def getImage(path, zoom):
                return OffsetImage(plt.imread(path), zoom=zoom, alpha=1)

            for team, x, y in zip(teams, attack_strength, defense_strength):
                logo_path = f"./imgs/{st.session_state['selected_league']}_logos/{team}.png"
                zoom = 0.3 if team in [home_team, away_team] else 0.15
                ab = AnnotationBbox(getImage(logo_path, zoom), (x, y), frameon=False)
                ax.add_artist(ab)

            ax.set_xlabel("Attack Strength (Higher is better)", labelpad=20, fontsize=12)
            ax.set_ylabel("Defense Strength (Lower is better)", labelpad=20, fontsize=12)
            ax.set_title(
                f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Team Attack and Defense Strength Over the Last {selected_week - 1} Weeks",
                fontsize=16,
                fontweight="bold",
                pad=70
            )

            home_adv_value = params.get("home_adv", 0) * 100
            if home_adv_value > 0:
                home_adv_text = f"The home team's goal expectation increases by approximately %{home_adv_value:.1f}"
            elif home_adv_value < 0:
                home_adv_text = f"The home team's goal expectation decreases by approximately %{abs(home_adv_value):.1f}"
            else:
                home_adv_text = ""

            if home_adv_text:
                ax.text(0.5, 1.12, home_adv_text, ha="center", va="center", fontsize=10, fontstyle="italic", color="gray", transform=ax.transAxes)

            ax.invert_yaxis()
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=10)
            add_footer(fig, x=0.96, y=-0.04, fontsize=8, extra_text=f"Results are based on the {selected_model} model.")

    elif selected_model == "Bradley-Terry":
        fig, ax = plt.subplots(figsize=(12, 10))

        colors = ["red" if team == home_team else "blue" if team == away_team else "gray" for team in model_df["Team"]]
        ax.barh(
            model_df["Team"],
            model_df["Rating"],
            color=colors,
            edgecolor=colors
        )

        fig.suptitle(
            f"{st.session_state['selected_league_original']} {st.session_state['selected_season_original']} Season - Matchweek {selected_week}\nTeam Strength Rankings and Home Win Probability",
            fontsize=18,
            fontweight="bold",
            y=1.00
        )

        ax.set_title(
            f"Home Win Probability for {home_team} vs. {away_team}: %{bt_prob * 100:.1f}",
            fontsize=14,
            fontweight="bold",
            pad=20
        )
        ax.set_xlabel("Rating", labelpad=20, fontsize=12)
        ax.set_ylabel("")
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        ax.invert_yaxis()
        add_footer(fig, x=0.98, y=-0.05, fontsize=8, extra_text=f"Results are based on the {selected_model} model.")
        plt.tight_layout()

    st.pyplot(fig)

def main(selected_model, selected_game, selected_week, plot_type, first_n_goals):
    try:

        match_data_df = get_data("match_data")

        match_data_df = match_data_df.rename(columns={
            "home_score_display":"home_team_goals",
            "away_score_display":"away_team_goals"
        })
        match_data_df["home_team_goals"] = pd.to_numeric(match_data_df["home_team_goals"], errors="coerce")
        match_data_df["away_team_goals"] = pd.to_numeric(match_data_df["away_team_goals"], errors="coerce")
        match_data_df = match_data_df[["week", "game_id", "home_team", "home_team_goals", "away_team", "away_team_goals", "status"]]
        match_data_df = match_data_df[match_data_df["status"] == "Ended"]

        home_team = selected_game.split("-")[0].strip()
        away_team = selected_game.split("-")[1].strip()

        if selected_model == "Dixon-Coles":
            params = solve_parameters_cached(match_data_df)
            model_df = dixon_coles_simulate_match_cached(params, home_team, away_team, max_goals=10)
            bt_prob = None
        elif selected_model == "Bradley-Terry":
            teams = np.sort(list(set(match_data_df["home_team"].unique()) | set(match_data_df["away_team"].unique())))
            bt_ratings, team_indices = solve_bt_ratings_cached(match_data_df, teams)
            bt_prob = bt_forecast_match_cached(bt_ratings, home_team, away_team, team_indices)
            model_df = pd.DataFrame({
                "Team": list(team_indices.keys()),
                "Rating": bt_ratings
            }).sort_values("Rating", ascending=False)
            model_df["Team"] = model_df["Team"].replace("home_field_advantage", "Ev Sahibi Avantajı")

        create_predictive_analytics_plot(
            model_df,
            selected_model,
            home_team,
            away_team,
            selected_week,
            plot_type,
            first_n_goals,
            bt_prob,
            params if selected_model == "Dixon-Coles" and plot_type == "Team Strength" else None
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