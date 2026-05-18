"""
db.py — Phase 1
Snowflake connection + table setup + insert helpers.

Run to test:
    python src/db.py
"""

import snowflake.connector
import pandas as pd
from config import (
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
    SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE,
)

# ── DDL ───────────────────────────────────────────────────────────────────────
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS lap_telemetry (
    id              INTEGER AUTOINCREMENT PRIMARY KEY,
    season          INTEGER,
    race            VARCHAR(100),
    driver          VARCHAR(10),
    lap             INTEGER,
    lap_time_s      FLOAT,
    tire            VARCHAR(20),
    tire_age        INTEGER,
    stint           INTEGER,
    position        INTEGER,
    deg_rate        FLOAT,
    stint_avg_pace  FLOAT,
    consistency     FLOAT,
    gap_to_leader   FLOAT,
    tire_code       INTEGER,
    inserted_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def get_connection():
    """Return an open Snowflake connection."""
    return snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
        warehouse=SNOWFLAKE_WAREHOUSE,
        role=SNOWFLAKE_ROLE,
    )


def setup_database():
    """Create database, schema, and table if they don't exist."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {SNOWFLAKE_DATABASE}")
        cur.execute(f"USE DATABASE {SNOWFLAKE_DATABASE}")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SNOWFLAKE_SCHEMA}")
        cur.execute(f"USE SCHEMA {SNOWFLAKE_SCHEMA}")
        cur.execute(CREATE_TABLE_SQL)
        print("✅ Snowflake table ready: lap_telemetry")


def insert_dataframe(df: pd.DataFrame):
    """
    Bulk-insert a features DataFrame into lap_telemetry.
    Skips rows already in Snowflake (same season + race + driver + lap).
    """
    cols = [
        "season", "race", "driver", "lap", "lap_time_s",
        "tire", "tire_age", "stint", "position",
        "deg_rate", "stint_avg_pace", "consistency",
        "gap_to_leader", "tire_code",
    ]
    df = df[cols].dropna(subset=["deg_rate", "consistency"])

    insert_sql = f"""
        INSERT INTO lap_telemetry
            ({", ".join(cols)})
        SELECT {", ".join(["%s"] * len(cols))}
        WHERE NOT EXISTS (
            SELECT 1 FROM lap_telemetry
            WHERE season = %s AND race = %s AND driver = %s AND lap = %s
        )
    """

    rows = [tuple(row) for row in df[cols].itertuples(index=False)]

    with get_connection() as conn:
        cur = conn.cursor()
        inserted = 0
        for row in rows:
            season, race, driver, lap = row[0], row[1], row[2], row[3]
            cur.execute(insert_sql, (*row, season, race, driver, lap))
            inserted += cur.rowcount

    print(f"✅ Inserted {inserted} new rows into Snowflake")
    return inserted


def query(sql: str) -> pd.DataFrame:
    """Run any SELECT and return a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Snowflake connection...")
    setup_database()

    # Verify with a simple query
    df = query("SELECT COUNT(*) as row_count FROM lap_telemetry")
    print(f"Current row count: {df['ROW_COUNT'].iloc[0]}")
    print("✅ Connection test passed")