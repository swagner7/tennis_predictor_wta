import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.predict import predict_match

def american_to_decimal(odds):
    return 1 + (abs(odds) / 100) if odds > 0 else 1 + (100 / abs(odds))

st.set_page_config(page_title="🎾 Tennis Diff‑Based Predictor", layout="wide")
st.title("🎾 Tennis Match Predictor")

data = pd.read_csv("data/processed/processed_matches_with_elo.csv", parse_dates=["Date"])
players = sorted(set(data['player1']).union(data['player2']))
surfaces = sorted(data['Surface'].dropna().unique())
series_list = sorted(data['Series'].dropna().unique())
rounds_list = sorted(data['Round'].dropna().unique())

# Selections
col1, col2 = st.columns(2)
with col1:
    player1 = st.selectbox("Player 1", players, index=0)
with col2:
    player2 = st.selectbox("Player 2", players, index=1)

surface = st.selectbox("Surface", surfaces)
series = st.selectbox("Series", series_list)
round_sel = st.selectbox("Round", rounds_list)

if player1 == player2:
    st.warning("Select two different players")
else:
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

    result = predict_match(player1, player2, surface, series, round_sel)
    prob = result['win_probability']
    p1_stats = result['player1_stats']
    p2_stats = result['player2_stats']

    if st.button("🔮 Predict Outcome"):
        st.success(f"Predicted Winner: **{result['predicted_winner']}**")
        st.metric(player1 + " Win Prob", f"{prob:.2%}")
        st.metric(player2 + " Win Prob", f"{(1-prob):.2%}")

        edge1 = prob - (1/odds1)
        edge2 = (1-prob) - (1/odds2)
        st.subheader("💸 Betting Recommendation")
        if edge1 > 0 and edge1 > edge2:
            st.success(f"Value Bet: {player1} (Edge {(edge1*100):.2f}%)")
        elif edge2 > 0:
            st.success(f"Value Bet: {player2} (Edge {(edge2*100):.2f}%)")
        else:
            st.info("No value bet detected")

    st.subheader("📊 Player Stats Comparison")
    stats_df = pd.DataFrame({
        'Stat': [
            'Elo', 'Surface Elo', 'Series Elo', 'Round Elo', 'H2H Wins',
            'Form (5)', 'Form (20)', 'Rank',
            'Experience', 'Days Since Last Match', 'Streak'
        ],
        player1: [
            p1_stats['Elo'], p1_stats['Surface_Elo'], p1_stats['Series_Elo'],
            p1_stats['Round_Elo'], p1_stats['h2h'],
            p1_stats['form_5'], p1_stats['form_20'], p1_stats['Rank'],
            p1_stats['experience'], p1_stats['days_since_last'], p1_stats['streak']
        ],
        player2: [
            p2_stats['Elo'], p2_stats['Surface_Elo'], p2_stats['Series_Elo'],
            p2_stats['Round_Elo'], p2_stats['h2h'],
            p2_stats['form_5'], p2_stats['form_20'], p2_stats['Rank'],
            p2_stats['experience'], p2_stats['days_since_last'], p2_stats['streak']
        ]
    })
    st.dataframe(stats_df.set_index('Stat'))

    st.subheader("📚 Head‑to‑Head Match History")
    h2h_hist = data[
        ((data['player1'] == player1) & (data['player2'] == player2)) |
        ((data['player1'] == player2) & (data['player2'] == player1))
    ].sort_values("Date", ascending=False)
    if h2h_hist.empty:
        st.write("No direct match history")
    else:
        display = h2h_hist[['Date', 'Tournament', 'Series', 'Surface', 'Round', 'player1', 'player2', 'player1_won']].copy()
        display['Winner'] = display.apply(lambda r: r['player1'] if r['player1_won'] == 1 else r['player2'], axis=1)
        st.dataframe(display.drop(columns=['player1_won']))
