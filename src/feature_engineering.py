import pandas as pd
import os
import numpy as np
from collections import defaultdict, deque
from elo import EloTracker

PROCESSED_DIR = os.path.join("data", "processed")
IN_PATH = os.path.join(PROCESSED_DIR, "processed_matches.csv")
OUT_PATH = os.path.join(PROCESSED_DIR, "processed_matches_with_elo.csv")

def remove_outliers_with_logging(df, col, iqr_factor=1.5):
    Q1, Q3 = df[col].quantile([.25, .75])
    IQR = Q3 - Q1
    lower, upper = Q1 - iqr_factor * IQR, Q3 + iqr_factor * IQR
    before = len(df)
    df = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f"Removed {before - len(df)} outliers from {col}")
    return df

def add_elo_and_features():
    df = pd.read_csv(IN_PATH, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    # Initialize trackers
    elo = EloTracker()
    tier_elo = EloTracker()
    round_elo = EloTracker()
    h2h = defaultdict(lambda: {"p1_wins": 0, "p2_wins": 0})
    match_counts = defaultdict(int)
    form_5 = defaultdict(lambda: deque(maxlen=5))
    form_20 = defaultdict(lambda: deque(maxlen=20))
    last_date = {}

    # Define feature columns
    cols = [
        "Elo_1", "Elo_2", "Surface_Elo_1", "Surface_Elo_2",
        "tier_elo_1", "tier_elo_2",
        "round_elo_1", "round_elo_2",
        "h2h_1", "h2h_2", "form_5_1", "form_5_2",
        "form_20_1", "form_20_2",
        "experience_1", "experience_2",
        "days_since_last_1", "days_since_last_2"
    ]
    features = {c: [] for c in cols}

    for _, row in df.iterrows():
        p1, p2 = row["player1"], row["player2"]
        surface, tier, rnd = row["Surface"], row["Tier"], row["Round"]
        winner = p1 if row["player1_won"] == 1 else p2

        # Record Elo before match
        e1, s1 = elo.get_elo(p1, surface)[0], elo.get_elo(p1, surface)[1]
        e2, s2 = elo.get_elo(p2, surface)[0], elo.get_elo(p2, surface)[1]
        se1 = tier_elo.get_elo(p1, tier)[1]
        se2 = tier_elo.get_elo(p2, tier)[1]
        re1 = round_elo.get_elo(p1, rnd)[1]
        re2 = round_elo.get_elo(p2, rnd)[1]

        features["Elo_1"].append(e1)
        features["Elo_2"].append(e2)
        features["Surface_Elo_1"].append(s1)
        features["Surface_Elo_2"].append(s2)
        features["tier_elo_1"].append(se1)
        features["tier_elo_2"].append(se2)
        features["round_elo_1"].append(re1)
        features["round_elo_2"].append(re2)

        # H2H
        h_key = tuple(sorted([p1, p2]))
        he = h2h[h_key]
        h1 = he["p1_wins"] if p1 < p2 else he["p2_wins"]
        h2 = he["p2_wins"] if p1 < p2 else he["p1_wins"]
        features["h2h_1"].append(h1)
        features["h2h_2"].append(h2)

        # Form
        features["form_5_1"].append(sum(form_5[p1]) / len(form_5[p1]) if form_5[p1] else 0.5)
        features["form_5_2"].append(sum(form_5[p2]) / len(form_5[p2]) if form_5[p2] else 0.5)
        features["form_20_1"].append(sum(form_20[p1]) / len(form_20[p1]) if form_20[p1] else 0.5)
        features["form_20_2"].append(sum(form_20[p2]) / len(form_20[p2]) if form_20[p2] else 0.5)

        # Experience & Days since last
        features["experience_1"].append(match_counts[p1])
        features["experience_2"].append(match_counts[p2])
        features["days_since_last_1"].append((row["Date"] - last_date.get(p1, row["Date"])).days)
        features["days_since_last_2"].append((row["Date"] - last_date.get(p2, row["Date"])).days)

        # Update Elo trackers
        elo.update_elo(p1, p2, winner, surface)
        tier_elo.update_elo(p1, p2, winner, tier)
        round_elo.update_elo(p1, p2, winner, rnd)

        # Update H2H counts
        if p1 < p2:
            h2h[h_key]["p1_wins" if winner == p1 else "p2_wins"] += 1
        else:
            h2h[h_key]["p2_wins" if winner == p1 else "p1_wins"] += 1

        # Update form
        form_5[p1].append(int(winner == p1))
        form_5[p2].append(int(winner == p2))
        form_20[p1].append(int(winner == p1))
        form_20[p2].append(int(winner == p2))

        # Update experience
        match_counts[p1] += 1
        match_counts[p2] += 1

        # Update last played dates
        last_date[p1] = row["Date"]
        last_date[p2] = row["Date"]

    # Assign computed features to dataframe
    for col, vals in features.items():
        df[col] = vals

    # Create difference-level features
    diff_map = {
        "elo_diff": ("Elo_1", "Elo_2"),
        "surface_elo_diff": ("Surface_Elo_1", "Surface_Elo_2"),
        "tier_elo_diff": ("tier_elo_1", "tier_elo_2"),
        "round_elo_diff": ("round_elo_1", "round_elo_2"),
        "h2h_diff": ("h2h_1", "h2h_2"),
        "form5_diff": ("form_5_1", "form_5_2"),
        "form20_diff": ("form_20_1", "form_20_2"),
        "experience_diff": ("experience_1", "experience_2"),
        "days_since_last_diff": ("days_since_last_1", "days_since_last_2"),
        "rankpts_diff": ("RankPts_1", "RankPts_2")
    }
    for new_col, (c1, c2) in diff_map.items():
        df[new_col] = df[c1] - df[c2]

    # # Remove outliers
    # for c in ["RankPts_1", "RankPts_2", "h2h_1", "h2h_2"]:
    #     df = remove_outliers_with_logging(df, c)

    df.to_csv(OUT_PATH, index=False)
    print("✅ Feature-enhanced dataset saved to:", OUT_PATH)


if __name__ == "__main__":
    add_elo_and_features()
