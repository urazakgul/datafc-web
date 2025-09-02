explanations = {
    "mean_distance": """
**How to interpret:**
This value shows the average distance between every pair of player positions on the field.
A higher number means the positions are more spread out from each other.
A lower number means the positions are closer together.
""",
    "horizontal_spread": """
**How to interpret:**
This value is the standard deviation of the x-coordinates (left-to-right positions) of all players.
A higher value means the positions are more widely spread across the width of the field.
A lower value means the positions are grouped more closely together horizontally.
""",
    "vertical_spread": """
**How to interpret:**
This value is the standard deviation of the y-coordinates (front-to-back positions) of all players.
A higher value means the positions are more spread out along the length of the field.
A lower value means the positions are grouped closer together vertically.
"""
}

pos_metric_explanations = {
    "Opponent Goal Distance": {
        "interpretation": (
            "Lower index → Average position is closer to the opponent goal  \n"
            "Higher index → Average position is farther from the opponent goal"
        ),
    },
    "Center Distance": {
        "interpretation": (
            "Lower index → Average position is closer to the pitch center  \n"
            "Higher index → Average position is farther from the pitch center"
        ),
    },
    "Diagonal Projection": {
        "interpretation": (
            "Lower index → Average position is closer to the origin of the length–width diagonal  \n"
            "Higher index → Average position is further along the diagonal"
        ),
    }
}