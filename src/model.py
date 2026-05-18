import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from config import FEATURES_CSV, MODEL_PKL

FEATURE_COLS = ["tire_code", "tire_age", "stint", "deg_rate", "consistency", "gap_to_leader"]
TARGET = "lap_time_s"

def train(df):
    X = df[FEATURE_COLS].dropna()
    y = df.loc[X.index, TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Model trained | MAE: {mae:.3f}s")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("\nFeature importances:")
    print(importances.sort_values(ascending=False).to_string())
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {MODEL_PKL}")
    return model

def generate_insights(df):
    insights = []
    deg = df.groupby("tire")["deg_rate"].mean().sort_values(ascending=False)
    for compound, rate in deg.items():
        if compound in ("SOFT", "MEDIUM", "HARD"):
            insights.append(f"{compound.capitalize()} tires degrade at {rate:+.3f}s/lap on average.")
    cons = df.groupby("driver")["consistency"].mean().sort_values()
    best = cons.index[0]
    insights.append(f"{best} is the most consistent driver (+-{cons[best]:.2f}s std dev per stint).")
    soft = df[df["tire"] == "SOFT"]["lap_time_s"].mean()
    hard = df[df["tire"] == "HARD"]["lap_time_s"].mean()
    if not (np.isnan(soft) or np.isnan(hard)):
        insights.append(f"Hard compound is {hard - soft:+.2f}s slower per lap vs soft on average.")
    return insights

if __name__ == "__main__":
    df = pd.read_csv(FEATURES_CSV)
    model = train(df)
    print("\n-- Insights --")
    for i in generate_insights(df):
        print(" *", i)
