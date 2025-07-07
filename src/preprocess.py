import pandas as pd
import os
import numpy as np

PROCESSED_DIR = os.path.join("data", "processed")
IN_PATH = os.path.join(PROCESSED_DIR, "all_matches.csv")
OUT_PATH = os.path.join(PROCESSED_DIR, "processed_matches.csv")

def preprocess():
    df = pd.read_csv(IN_PATH, parse_dates=['Date'])

    # Keep only completed matches
    df = df[df['Comment'] == 'Completed'].copy()
    df['Winner'] = df['Winner'].str.strip()
    df['Loser'] = df['Loser'].str.strip()

    # Shuffle player1/player2 assignment
    np.random.seed(42)
    rows = []
    for _, r in df.iterrows():
        common = {
            'WTA': r.WTA,
            'Location': r.Location,
            'Tournament': r.Tournament,
            'Date': r.Date,
            'Tier': r.Tier,
            'Court': r.Court,
            'Surface': r.Surface,
            'Round': r.Round,
            'Best_of': r['Best of']
        }

        if np.random.rand() < 0.5:
            rows.append({**common,
                         'player1': r.Winner,
                         'player2': r.Loser,
                         'Rank_1': r.WRank,
                         'Rank_2': r.LRank,
                         'Odds_1': r.AvgW,
                         'Odds_2': r.AvgL,
                         'player1_won': 1})
        else:
            rows.append({**common,
                         'player1': r.Loser,
                         'player2': r.Winner,
                         'Rank_1': r.LRank,
                         'Rank_2': r.WRank,
                         'Odds_1': r.AvgL,
                         'Odds_2': r.AvgW,
                         'player1_won': 0})

    final_df = pd.DataFrame(rows)

    # Drop NaNs and duplicates
    final_df.dropna(inplace=True)
    final_df.drop_duplicates(inplace=True)

    # Sort by Date
    final_df.sort_values('Date', inplace=True)

    final_df.to_csv(OUT_PATH, index=False)
    print(f"✅ Processed data saved to {OUT_PATH}")

if __name__ == "__main__":
    preprocess()
