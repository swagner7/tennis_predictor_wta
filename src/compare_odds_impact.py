import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/processed/processed_matches_with_elo.csv")
TARGET = "player1_won"

ALL_FEATURES = [
    'Elo_1', 'Elo_2', 'Surface_Elo_1', 'Surface_Elo_2',
    'h2h_1', 'h2h_2', 'form_5_1', 'form_5_2',
    'form_20_1', 'form_20_2', 'Rank_1', 'Rank_2',
    'Odds_1', 'Odds_2'
]

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

def evaluate(X, y):
    cutoff = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    results = {}
    for name, model in MODELS.items():
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, (probs > 0.5).astype(int))
        auc = roc_auc_score(y_test, probs)

        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        elif hasattr(model, "coef_"):
            fi = model.coef_[0]
        else:
            fi = np.zeros(X.shape[1])

        results[name] = {
            "accuracy": acc,
            "auc": auc,
            "importances": pd.Series(fi, index=X.columns)
        }

    return results

# Evaluate with odds
X_all = df[ALL_FEATURES]
y = df[TARGET]
results_with = evaluate(X_all, y)

# Evaluate without odds
X_no_odds = df[[f for f in ALL_FEATURES if "Odds" not in f]]
results_without = evaluate(X_no_odds, y)

# Plot comparison
for model_name in MODELS.keys():
    fi_with = results_with[model_name]["importances"]
    fi_without = results_without[model_name]["importances"]
    all_features = sorted(set(fi_with.index).union(fi_without.index))

    df_plot = pd.DataFrame({
        "Feature": all_features,
        "With Odds": [fi_with.get(f, 0) for f in all_features],
        "Without Odds": [fi_without.get(f, 0) for f in all_features]
    })

    df_plot = df_plot.set_index("Feature")
    df_plot = df_plot.sort_values("With Odds", ascending=False)

    df_plot.plot(kind="barh", figsize=(10, 8))
    plt.title(f"Feature Importances: {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"figures/feature_importance_comparison_{model_name}.png")
    plt.close()
    print(f"✅ Saved: figures/feature_importance_comparison_{model_name}.png")

# Summary table
summary = []
for name in MODELS:
    summary.append({
        "Model": name,
        "WithOdds_ACC": results_with[name]["accuracy"],
        "WithOdds_AUC": results_with[name]["auc"],
        "NoOdds_ACC": results_without[name]["accuracy"],
        "NoOdds_AUC": results_without[name]["auc"]
    })

pd.DataFrame(summary).to_csv("models/odds_vs_no_odds_results.csv", index=False)
print("✅ Saved: models/odds_vs_no_odds_results.csv")
