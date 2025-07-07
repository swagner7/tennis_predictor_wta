import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.predict import predict_match

def american_to_decimal(odds):
    return 1 + (abs(odds) / 100) if odds > 0 else 1 + (100 / abs(odds))

def calc_bet(line, win_pct):
    # Convert American to Decimal
    if line > 0:
        decimal_odds = line / 100 + 1
    else:
        odds = 1 - (100 / line)
    num = pow((win_pct * odds - 1), 3)
    den = (odds - 1) * (pow((win_pct * odds - 1), 2) + (pow(odds, 2) * sigma))
    f_star = round(100 * (num / den), 2)

    if kelly_fraction > 0:
        return f"{round(100 * kelly_fraction, 2)}% of bankroll"
    else:
        return "Bet not advised"


st.set_page_config(page_title="🎾 Tennis Diff‑Based Predictor", layout="wide")
st.title("🎾 Tennis Match Predictor")

data = pd.read_csv("data/processed/processed_matches_with_elo.csv", parse_dates=["Date"])
players = sorted(set(data['player1']).union(data['player2']))
surfaces = sorted(data['Surface'].dropna().unique())
tier_options = sorted(data['Tier'].dropna().unique())
rounds = sorted(data['Round'].dropna().unique())

# Player + Surface + tier + Round selection
col1, col2 = st.columns(2)
with col1:
    player1 = st.selectbox("Player 1", players, index=0)
with col2:
    player2 = st.selectbox("Player 2", players, index=1)
surface = st.selectbox("Surface", surfaces)
tier = st.selectbox("Tier", tier_options)
round_name = st.selectbox("Round", rounds)

if player1 == player2:
    st.warning("Select two different players")
else:
    try:
        st.subheader("💵 Enter American Odds (for betting only)")
        c1, c2 = st.columns(2)
        with c1:
            odds1_us = st.number_input(f"{player1} Odds (e.g. +150, -200)", value=+150)
        with c2:
            odds2_us = st.number_input(f"{player2} Odds (e.g. +150, -200)", value=-150)

        odds1 = american_to_decimal(odds1_us)
        odds2 = american_to_decimal(odds2_us)

        st.subheader("📈 Implied Win Probabilities")
        ic1, ic2 = st.columns(2)
        ic1.metric(player1, f"{(1/odds1):.2%}")
        ic2.metric(player2, f"{(1/odds2):.2%}")

        result = predict_match(player1, player2, surface, tier, round_name)
        prob = result['win_probability']
        p1_stats = result['player1_stats']
        p2_stats = result['player2_stats']

        if st.button("🔮 Predict Outcome"):
            st.success(f"Predicted Winner: **{result['predicted_winner']}**")

            st.subheader("🧠 ML Win Probabilities")
            ml1, ml2 = st.columns(2)
            ml1.metric(f"{player1}", f"{prob:.2%}")
            ml2.metric(f"{player2}", f"{(1 - prob):.2%}")

            st.subheader("💸 Betting Recommendation (Bankroll Strategy)")
            reco1 = calc_bet(odds1_us, prob)
            reco2 = calc_bet(odds2_us, 1 - prob)
            c1, c2 = st.columns(2)
            c1.markdown(f"**{player1}**: {reco1}")
            c2.markdown(f"**{player2}**: {reco2}")


        st.subheader("📊 Player Stats Comparison")
        stats_df = pd.DataFrame({
            'Stat': [
                "Global Elo", "Surface Elo", "Tier Elo", "Round Elo",
                "Head-to-Head", "Form (5)", "Form (20)", "Experience",
                "Days Since Last", "Streak", "Rank"
            ],
            player1: [
                p1_stats['Elo'], p1_stats['Surface_Elo'], p1_stats['Tier_Elo'], p1_stats['Round_Elo'],
                p1_stats['h2h'], p1_stats['form_5'], p1_stats['form_20'], p1_stats['experience'],
                p1_stats['days_since_last'], p1_stats['streak'], p1_stats['Rank']
            ],
            player2: [
                p2_stats['Elo'], p2_stats['Surface_Elo'], p2_stats['Tier_Elo'], p2_stats['Round_Elo'],
                p2_stats['h2h'], p2_stats['form_5'], p2_stats['form_20'], p2_stats['experience'],
                p2_stats['days_since_last'], p2_stats['streak'], p2_stats['Rank']
            ]
        })
        st.dataframe(stats_df.set_index("Stat"))

        st.subheader("📚 Head‑to‑Head Match History")
        h2h_hist = data[
            ((data['player1'] == player1) & (data['player2'] == player2)) |
            ((data['player1'] == player2) & (data['player2'] == player1))
        ].sort_values("Date", ascending=False)
        if h2h_hist.empty:
            st.write("No direct match history")
        else:
            display = h2h_hist[['Date', 'Tournament', 'Surface', 'player1', 'player2', 'player1_won']].copy()
            display['Winner'] = display.apply(lambda r: r['player1'] if r['player1_won'] == 1 else r['player2'], axis=1)
            st.dataframe(display.drop(columns=['player1_won']))

    except ValueError as e:
        st.error(str(e))
