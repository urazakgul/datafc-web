PLOT_STYLE = "fivethirtyeight"

event_colors = {
    "pass": "#0074D9",
    "goal": "#FF4136",
    "free-kick": "#2ECC40",
    "clearance": "#B10DC9",
    "ball-movement": "#FF851B",
    "corner": "#FFDC00",
    "post": "#FF69B4",
    "save": "#7FDBFF",
    "miss": "#AAAAAA"
}

shot_colors = {
    "save": "blue",
    "miss": "orange",
    "block": "purple",
    "post": "black"
}

match_performances = [
    "Ball Possession",
    "Pass Success Rate",
    "Big Chances Scored/Missed",
    "Shot Accuracy",
    "Shot Ratio Inside/Outside Box",
    "Touches in Penalty Area",
    "Final Third Entries",
    "Successful Actions in Final Third",
    "Difference Between Fouls Committed and Suffered",
    "Cards Per Foul",
    "Accurate Long Pass Rate",
    "Accurate Cross Rate"
]

match_performance_binary = [
    "Final third phase",
    "Long balls",
    "Crosses",
    "Ground duels",
    "Aerial duels",
    "Dribbles"
]

match_performance_posneg = {
    "Positive": [
        "Ball possession",
        "Expected goals",
        "Big chances",
        "Total shots",
        "Goalkeeper saves",
        "Corner kicks",
        "Passes",
        "Tackles",
        "Free kicks",
        "Shots on target",
        "Blocked shots",
        "Shots inside box",
        "Shots outside box",
        "Through balls",
        "Fouled in final third",
        "Accurate passes",
        "Throw-ins",
        "Final third entries",
        "Long balls",
        "Crosses",
        "Duels",
        "Ground duels",
        "Aerial duels",
        "Dribbles",
        "Tackles won",
        "Total tackles",
        "Interceptions",
        "Recoveries",
        "Clearances",
        "Total saves",
        "Goals prevented",
        "Goal kicks",
        "Big chances scored",
        "Touches in penalty area",
        "Final third phase",
        "Big saves",
        "High claims",
        "Punches",
        "Penalty saves"
    ],
    "Negative": [
        "Fouls",
        "Yellow cards",
        "Shots off target",
        "Big chances missed",
        "Offsides",
        "Dispossessed",
        "Red cards",
        "Errors lead to a shot",
        "Errors lead to a goal"
    ]
}

game_stats_group_name = [
    "Match overview",
    "Shots",
    "Attack",
    "Passes",
    "Duels",
    "Defending",
    "Goalkeeping"
]