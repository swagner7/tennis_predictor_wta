import pandas as pd
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED = os.path.join("data", "processed", "processed_matches_with_elo.csv")
MODEL_PATH = os.path.join("models", "best_model.pkl")
CSV_RESULTS_PATH = os.path.join("models", "training_results.csv")
os.makedirs("models", exist_ok=True)

FEATURES = [
    "elo_diff", "surface_elo_diff", "tier_elo_diff", "round_elo_diff",
    "h2h_diff", "form5_diff", "form20_diff",
    "experience_diff", "days_since_last_diff", "rankpts_diff"
]
TARGET = "player1_won"

def get_probs(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten()
    else:
        return model.predict(X).astype(float)

def train():
    df = pd.read_csv(PROCESSED, parse_dates=["Date"])
    df = df.sort_values("Date")
    cutoff = int(len(df) * 0.8)

    X_train = df.iloc[:cutoff][FEATURES]
    y_train = df.iloc[:cutoff][TARGET]
    X_test = df.iloc[cutoff:][FEATURES]
    y_test = df.iloc[cutoff:][TARGET]

    base_models = {
        "LogisticRegression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
        "RidgeClassifier": Pipeline([("scale", StandardScaler()), ("model", RidgeClassifier())]),
        "SVC": Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True))]),
        "KNN": Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())]),
        "MLP": Pipeline([("scale", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(64,), max_iter=500))]),
        "RandomForest": Pipeline([("scale", StandardScaler()), ("model", RandomForestClassifier(n_estimators=100))]),
        "ExtraTrees": Pipeline([("scale", StandardScaler()), ("model", ExtraTreesClassifier(n_estimators=100))]),
        "XGBoost": Pipeline([("scale", StandardScaler()), ("model", XGBClassifier(use_label_encoder=False, eval_metric="logloss") )]),
        "LightGBM": Pipeline([("scale", StandardScaler()), ("model", LGBMClassifier())]),
        "GradientBoosting": Pipeline([("scale", StandardScaler()), ("model", GradientBoostingClassifier())]),
        "HistGradientBoosting": Pipeline([("scale", StandardScaler()), ("model", HistGradientBoostingClassifier())])
    }

    param_grids = {
        "LogisticRegression": {"model__C": [0.01, 0.1, 1, 10]},
        "RidgeClassifier": {"model__alpha": [0.1, 1.0, 10.0]},
        "SVC": {"model__C": [0.1, 1, 10]},
        "KNN": {"model__n_neighbors": [3, 5, 10, 20]},
        "MLP": {"model__hidden_layer_sizes": [(32,), (64,)]},
        "RandomForest": {"model__model__n_estimators": [100, 200], "model__model__max_depth": [None, 10, 20]},
        "XGBoost": {"model__model__n_estimators": [100, 200], "model__model__max_depth": [3, 6]}
    }

    results = []
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        probs = get_probs(model, X_test)
        acc = accuracy_score(y_test, (probs > 0.5).astype(int))
        auc = roc_auc_score(y_test, probs)
        results.append({"Model": name, "Accuracy": acc, "AUC": auc})
        print(f"{name}: ACC={acc:.4f}, AUC={auc:.4f}")

    pd.DataFrame(results).sort_values("Accuracy", ascending=False).to_csv(CSV_RESULTS_PATH, index=False)

    top3 = pd.DataFrame(results).nlargest(3, "Accuracy")["Model"].tolist()
    best, best_score = None, 0
    for name in top3:
        print(f"Tuning {name}...")
        grid = GridSearchCV(base_models[name], param_grids.get(name, {}), scoring="accuracy", cv=3)
        grid.fit(X_train, y_train)
        score = grid.best_score_
        print(f"  {name} CV accuracy: {score:.4f}")
        if score > best_score:
            best_score = score
            best = grid.best_estimator_

    joblib.dump(best, MODEL_PATH)
    print(f"Saved best model with CV accuracy {best_score:.4f} to {MODEL_PATH}")

if __name__ == "__main__":
    train()
