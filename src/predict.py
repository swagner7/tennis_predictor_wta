import pandas as pd
import joblib
import os
import numpy as np

MODEL_PATH = os.path.join("models", "best_model.pkl")
DATA_PATH = os.path.join("data", "processed", "processed_matches_with_elo.csv")

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH, parse_dates=["Date"])

# Only diff features now
FEATURE_COLS = [
    "elo_diff", "surface_elo_diff", "tier_elo_diff", "round_elo_diff",
    "h2h_diff", "form5_diff", "form20_diff",
    "experience_diff", "days_since_last_diff", "rankpts_diff"
]

def get_latest_player_features(player, opponent, surface, tier, round_name, days_since_last):
    recent = data[(data['player1'] == player) | (data['player2'] == player)].sort_values("Date", ascending=False)
    if recent.empty:
        raise ValueError(f"No match history for {player}")

    row = recent.iloc[0]
    flip = row['player2'] == player
    def col(colname):
        return row[f"{colname}_2"] if flip else row[f"{colname}_1"]

    # Context-specific Elos
    def get_context_elo(context_col, value, elo_col_prefix):
        matches = data[((data['player1'] == player) | (data['player2'] == player)) & (data[context_col] == value)]
        if matches.empty:
            return 1500
        latest = matches.sort_values("Date", ascending=False).iloc[0]
        flip = latest['player2'] == player
        return latest[f"{elo_col_prefix}_2"] if flip else latest[f"{elo_col_prefix}_1"]

    surface_elo = get_context_elo("Surface", surface, "Surface_Elo")
    tier_elo = get_context_elo("Tier", tier, "tier_elo")
    round_elo = get_context_elo("Round", round_name, "round_elo")

    h2h_matches = data[
        ((data['player1'] == player) & (data['player2'] == opponent)) |
        ((data['player1'] == opponent) & (data['player2'] == player))
    ]
    h2h_1 = sum((h2h_matches['player1'] == player) & (h2h_matches['player1_won'] == 1)) + \
            sum((h2h_matches['player2'] == player) & (h2h_matches['player1_won'] == 0))

    return {
        "elo": col("Elo"),
        "surface_elo": surface_elo,
        "tier_elo": tier_elo,
        "round_elo": round_elo,
        "h2h": h2h_1,
        "form_5": col("form_5"),
        "form_20": col("form_20"),
        "rankpts": col("RankPts"),
        "experience": col("experience"),
        "days_since_last": days_since_last
    }

def predict_match(player1, player2, surface, tier, round_name, _date_of_match=None,
                  days_since_last_1=None, days_since_last_2=None):
    original_order = (player1, player2)
    if player2 < player1:
        player1, player2 = player2, player1
        days_since_last_1, days_since_last_2 = days_since_last_2, days_since_last_1
        flipped = True
    else:
        flipped = False

    p1 = get_latest_player_features(player1, player2, surface, tier, round_name, days_since_last_1)
    p2 = get_latest_player_features(player2, player1, surface, tier, round_name, days_since_last_2)

    diff_vector = [
        p1['elo'] - p2['elo'],
        p1['surface_elo'] - p2['surface_elo'],
        p1['tier_elo'] - p2['tier_elo'],
        p1['round_elo'] - p2['round_elo'],
        p1['h2h'] - p2['h2h'],
        p1['form_5'] - p2['form_5'],
        p1['form_20'] - p2['form_20'],
        p1['experience'] - p2['experience'],
        p1['days_since_last'] - p2['days_since_last'],
        p1['rankpts'] - p2['rankpts']
    ]

    X = np.array(diff_vector).reshape(1, -1)
    prob = model.predict_proba(X)[0][1]

    if flipped:
        prob = 1 - prob
        player1, player2 = player2, player1
        p1, p2 = p2, p1

    return {
        "win_probability": prob,
        "predicted_winner": player1 if prob > 0.5 else player2,
        "player1_stats": p1,
        "player2_stats": p2
    }
