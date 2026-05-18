import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import fastf1
import pandas as pd
from config import DATA_RAW, DATA_PROCESSED, LAPS_CSV, SEASONS_TO_PULL

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(DATA_RAW))

def get_all_races(year):
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        races = schedule[schedule["EventFormat"] != "testing"]["EventName"].tolist()
        print(f"  Found {len(races)} races in {year}")
        return races
    except Exception as e:
        print(f"  WARNING Could not fetch schedule for {year}: {e}")
        return []

def fetch_race(year, race_name):
    try:
        print(f"    Fetching {year} {race_name}...")
        session = fastf1.get_session(year, race_name, "R")
        session.load(telemetry=False, weather=False, messages=False)

        laps = session.laps[
            ["Driver", "LapNumber", "LapTime", "Compound",
             "TyreLife", "Stint", "Position"]
        ].copy()

        laps.dropna(subset=["LapTime", "Compound"], inplace=True)
        laps["lap_time_s"] = laps["LapTime"].dt.total_seconds()

        medians = laps.groupby("Driver")["lap_time_s"].transform("median")
        laps = laps[laps["lap_time_s"] < medians * 1.10].copy()

        laps = laps.rename(columns={
            "Driver": "driver", "LapNumber": "lap", "Compound": "tire",
            "TyreLife": "tire_age", "Stint": "stint", "Position": "position",
        })
        laps["race"]   = race_name
        laps["season"] = year

        return laps[["season", "race", "driver", "lap",
                      "lap_time_s", "tire", "tire_age", "stint", "position"]]
    except Exception as e:
        print(f"    WARNING Skipped {year} {race_name}: {e}")
        return None

def build_dataset():
    frames = []
    for year in SEASONS_TO_PULL:
        print(f"\nSeason {year}")
        for race in get_all_races(year):
            df = fetch_race(year, race)
            if df is not None and len(df) > 0:
                frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(LAPS_CSV, index=False)
    print(f"\nDone. Saved {len(df):,} laps to {LAPS_CSV}")
    return df

if __name__ == "__main__":
    df = build_dataset()
    print(f"Shape:   {df.shape}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Races:   {df['race'].nunique()} total")
