import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from config import LAPS_CSV, FEATURES_CSV

def load_laps():
    return pd.read_csv(LAPS_CSV)

def add_tire_degradation(df):
    def stint_slope(g):
        if len(g) < 3:
            return pd.Series(np.nan, index=g.index)
        try:
            x = g["tire_age"].values.astype(float)
            y = g["lap_time_s"].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                return pd.Series(np.nan, index=g.index)
            slope = np.polyfit(x[mask], y[mask], 1)[0]
            return pd.Series(slope, index=g.index)
        except Exception:
            return pd.Series(np.nan, index=g.index)

    df["deg_rate"] = (
        df.groupby(["season", "race", "driver", "stint"], group_keys=False)
          .apply(stint_slope)
    )
    return df

def add_stint_pace(df):
    df["stint_avg_pace"] = (
        df.groupby(["season", "race", "driver", "stint"])["lap_time_s"]
          .transform("mean")
    )
    return df

def add_consistency(df):
    df["consistency"] = (
        df.groupby(["season", "race", "driver", "stint"])["lap_time_s"]
          .transform("std")
    )
    return df

def add_relative_pace(df):
    race_best = (
        df.groupby(["season", "race", "lap"])["lap_time_s"]
          .transform("min")
    )
    df["gap_to_leader"] = df["lap_time_s"] - race_best
    return df

def add_tire_encoded(df):
    order = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
    df["tire_code"] = df["tire"].map(order).fillna(-1).astype(int)
    return df

def build_features(df):
    print("Building features...")
    df = add_tire_degradation(df)
    df = add_stint_pace(df)
    df = add_consistency(df)
    df = add_relative_pace(df)
    df = add_tire_encoded(df)
    df.dropna(subset=["deg_rate", "consistency"], inplace=True)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"Done. Saved {len(df):,} rows to {FEATURES_CSV}")
    return df

if __name__ == "__main__":
    df = load_laps()
    df = build_features(df)
    print(df[["driver", "tire", "tire_age", "deg_rate", "consistency", "gap_to_leader"]].head(15).to_string())
