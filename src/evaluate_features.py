import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance

PROCESSED = "data/processed/processed_matches_with_elo.csv"
MODEL_PATH = "models/best_model.pkl"
PLOT_DIR = "reports/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET = "player1_won"

# Load dataset and model
df = pd.read_csv(PROCESSED, parse_dates=["Date"])
model = joblib.load(MODEL_PATH)

# Directly specify the active features
FEATURES = [
    "elo_diff", "surface_elo_diff", "tier_elo_diff", "round_elo_diff",
    "h2h_diff", "form5_diff", "form20_diff",
    "experience_diff", "days_since_last_diff", "rankpts_diff"
]

# Drop rows with missing values in those features
df = df.dropna(subset=FEATURES)
X = df[FEATURES]
y = df[TARGET]

# 1️⃣ Plot distributions
for col in FEATURES:
    plt.figure(figsize=(6, 4))
    sns.histplot(X[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{col}_dist.png"))
    plt.close()

# 2️⃣ Plot correlations
corr = X.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"))
plt.close()

# 3️⃣ Feature importance
base = model.named_steps["model"] if hasattr(model, "named_steps") else model
try:
    importances = base.feature_importances_
    source = "model"
except AttributeError:
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importances = perm.importances_mean
    source = "permutation"

# Ensure proper mapping
if len(importances) != len(FEATURES):
    raise ValueError(f"Mismatch: {len(importances)} importances vs {len(FEATURES)} features")

# 4️⃣ Plot feature importances
imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances}).sort_values("Importance", ascending=False)
print(imp_df)

plt.figure(figsize=(10, 0.5 * len(imp_df)))
sns.barplot(data=imp_df, x="Importance", y="Feature", palette="viridis", dodge=False)
plt.title(f"Feature Importance ({source})")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"))
plt.close()

print("✅ Evaluation complete! All plots saved to:", PLOT_DIR)
