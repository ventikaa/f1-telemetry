import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np
import pandas as pd
from config import MODEL_PKL

FEATURE_COLS = ["tire_code", "tire_age", "stint", "deg_rate", "consistency", "gap_to_leader"]
TIRE_CODES   = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
PIT_COST_S   = 22.0

def load_model():
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)

def simulate_strategy(model, race_laps, pit_lap, compound_1="SOFT", compound_2="HARD",
                      deg_rate=0.05, consistency=0.25, gap_to_leader=0.0, noise=0.0):
    total_time = 0.0
    lap_times  = []
    for lap in range(1, race_laps + 1):
        if lap <= pit_lap:
            stint, tire_code, tire_age = 1, TIRE_CODES[compound_1], lap
        else:
            stint, tire_code, tire_age = 2, TIRE_CODES[compound_2], lap - pit_lap
        features = pd.DataFrame([{
            "tire_code": tire_code, "tire_age": tire_age, "stint": stint,
            "deg_rate": deg_rate, "consistency": consistency, "gap_to_leader": gap_to_leader,
        }], columns=FEATURE_COLS)
        pred = model.predict(features)[0]
        if noise > 0:
            pred += np.random.normal(0, noise)
        if lap == pit_lap + 1:
            total_time += PIT_COST_S
        total_time += pred
        lap_times.append({"lap": lap, "lap_time_s": pred,
                          "tire": compound_1 if lap <= pit_lap else compound_2})
    return {"pit_lap": pit_lap, "compound_1": compound_1, "compound_2": compound_2,
            "total_s": total_time, "lap_times": pd.DataFrame(lap_times)}

def find_optimal_window(model, race_laps=57, compound_1="SOFT", compound_2="HARD",
                        pit_range=(10, 35), **kwargs):
    results = []
    for pit_lap in range(pit_range[0], pit_range[1] + 1):
        r = simulate_strategy(model, race_laps, pit_lap, compound_1, compound_2, **kwargs)
        results.append({"pit_lap": pit_lap, "total_s": r["total_s"]})
    df = pd.DataFrame(results).sort_values("total_s").reset_index(drop=True)
    df["delta_s"] = df["total_s"] - df["total_s"].iloc[0]
    return df

if __name__ == "__main__":
    model = load_model()
    print("-- Optimal window (SOFT -> HARD, 57 laps) --")
    window = find_optimal_window(model, race_laps=57)
    print(window.head(10).to_string(index=False))
    print(f"\nOptimal pit lap: {int(window.iloc[0]['pit_lap'])}")
