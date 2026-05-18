import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR       = Path(__file__).parent.parent
DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
LAPS_CSV       = DATA_PROCESSED / "laps.csv"
FEATURES_CSV   = DATA_PROCESSED / "features.csv"
MODEL_PKL      = DATA_PROCESSED / "model.pkl"

SEASONS_TO_PULL = list(range(2018, 2026))

SNOWFLAKE_ACCOUNT   = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER      = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD  = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_DATABASE  = os.getenv("SNOWFLAKE_DATABASE", "F1_DB")
SNOWFLAKE_SCHEMA    = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
SNOWFLAKE_ROLE      = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
