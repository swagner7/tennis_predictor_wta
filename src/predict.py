import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join("models", "best_model.pkl")
DATA_PATH = os.path.join("data", "processed", "processed_matches_with_elo.csv")

# Load once on import
model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH, parse_dates=["Date"])

FEATURE_COLS = [
    "elo_diff", "surface_elo_diff", "series_elo_diff", "round_elo_diff",
    "h2h_diff", "form5_diff", "form20_diff", "rank_diff",
    "experience_diff", "days_since_last_diff", "streak_diff"
]


def get_latest_player_features(player, opponent, surface, series, round_name):
    # Sort descending to get most recent
    recent = data[(data['player1'] == player) | (data['player2'] == player)].sort_values("Date", ascending=False)

    if recent.empty:
        raise ValueError(f"No match history for {player}")

    row = recent.iloc[0]
    flip = row['player2'] == player

    def col(colname):
        return row[f"{colname}_2"] if flip else row[f"{colname}_1"]

    # Pull most recent surface-elo
    surf_elo = 1500
    surface_matches = data[((data['player1'] == player) | (data['player2'] == player)) & (data["Surface"] == surface)]
    if not surface_matches.empty:
        srow = surface_matches.sort_values("Date", ascending=False).iloc[0]
        surf_elo = srow["Surface_Elo_2"] if srow["player2"] == player else srow["Surface_Elo_1"]

    # Series Elo
    series_elo = 1500
    series_matches = data[((data['player1'] == player) | (data['player2'] == player)) & (data["Series"] == series)]
    if not series_matches.empty:
        srow = series_matches.sort_values("Date", ascending=False).iloc[0]
        series_elo = srow["series_elo_2"] if srow["player2"] == player else srow["series_elo_1"]

    # Round Elo
    round_elo = 1500
    round_matches = data[((data['player1'] == player) | (data['player2'] == player)) & (data["Round"] == round_name)]
    if not round_matches.empty:
        rrow = round_matches.sort_values("Date", ascending=False).iloc[0]
        round_elo = rrow["round_elo_2"] if rrow["player2"] == player else rrow["round_elo_1"]

    # H2H
    h2h_matches = data[
        ((data['player1'] == player) & (data['player2'] == opponent)) |
        ((data['player1'] == opponent) & (data['player2'] == player))
    ]
    h2h_1 = sum((h2h_matches['player1'] == player) & (h2h_matches['player1_won'] == 1)) + \
            sum((h2h_matches['player2'] == player) & (h2h_matches['player1_won'] == 0))

    return {
        "Elo": col("Elo"),
        "Surface_Elo": surf_elo,
        "Series_Elo": series_elo,
        "Round_Elo": round_elo,
        "h2h": h2h_1,
        "form_5": col("form_5"),
        "form_20": col("form_20"),
        "Rank": col("Rank"),
        "experience": col("experience"),
        "days_since_last": col("days_since_last"),
        "streak": col("streak"),
    }


def predict_match(player1, player2, surface, series, round_name):
    p1 = get_latest_player_features(player1, player2, surface, series, round_name)
    p2 = get_latest_player_features(player2, player1, surface, series, round_name)

    # Diffs for prediction model
    X = [[
        p1['Elo'] - p2['Elo'],
        p1['Surface_Elo'] - p2['Surface_Elo'],
        p1['Series_Elo'] - p2['Series_Elo'],
        p1['Round_Elo'] - p2['Round_Elo'],
        p1['h2h'] - p2['h2h'],
        p1['form_5'] - p2['form_5'],
        p1['form_20'] - p2['form_20'],
        p1['Rank'] - p2['Rank'],
        p1['experience'] - p2['experience'],
        p1['days_since_last'] - p2['days_since_last'],
        p1['streak'] - p2['streak'],
    ]]

    prob = model.predict_proba(X)[0][1]
    return {
        "win_probability": prob,
        "predicted_winner": player1 if prob > 0.5 else player2,
        "player1_stats": p1,
        "player2_stats": p2
    }
