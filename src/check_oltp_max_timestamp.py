"""Print the max `timestamp` from the OLTP `movie_ratings` table.

Used before `feast materialize` to set the materialization checkpoint:

    MATERIALIZE_CHECKPOINT_TIME=$(uv run src/check_oltp_max_timestamp.py \
        | awk -F'<ts>|</ts>' '{print $2}')
    uv run feast materialize 2010-01-01T00:00:00 "$MATERIALIZE_CHECKPOINT_TIME"

Ported from the reference `src/check_oltp_max_timestamp.py`. Differences:
  - connection string built from .env (PG_* vars) instead of hardcoded
  - table name from PG_TABLE (default `movie_ratings`) instead of `reviews`
  - sslmode=require for non-local hosts (mirrors notebooks/002 convention)

Run from the project root:  uv run src/check_oltp_max_timestamp.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine

logger.remove()
logger.add(sys.stderr, level="INFO")

# .env lives at the project root (one level up from src/).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _build_engine():
    """Build a SQLAlchemy engine from PG_* env vars (defaults mirror 002)."""
    pg_user = os.getenv("PG_USER") or "postgres"
    pg_password = os.getenv("PG_PASSWORD") or "postgres"
    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = os.getenv("PG_PORT", "5435")
    pg_db = os.getenv("PG_DB", "raw_data")

    # Require SSL for real RDS; disable for local Docker Postgres.
    is_local = pg_host in ("localhost", "127.0.0.1", "0.0.0.0")
    sslmode = "disable" if is_local else "require"

    conn_str = (
        f"postgresql+psycopg2://{pg_user}:{pg_password}"
        f"@{pg_host}:{pg_port}/{pg_db}?sslmode={sslmode}"
    )
    logger.info(f"connecting to {pg_host}:{pg_port}/{pg_db} (local={is_local}, sslmode={sslmode})")
    return create_engine(conn_str)


def get_curr_oltp_max_timestamp():
    """Return the max `timestamp` from the OLTP table as a UTC tz-aware datetime."""
    schema = os.getenv("PG_SCHEMA", "public")
    table_name = os.getenv("PG_TABLE", "movie_ratings")
    engine = _build_engine()

    query = f"SELECT max(timestamp) AS max_timestamp FROM {schema}.{table_name};"
    max_timestamp = pd.read_sql(query, engine)["max_timestamp"].iloc[0]

    if pd.notnull(max_timestamp):
        # OLTP stores naive datetime (002 wrote epoch->naive UTC); make it tz-aware.
        max_timestamp = pd.to_datetime(max_timestamp)
        if max_timestamp.tzinfo is None:
            max_timestamp = max_timestamp.tz_localize("UTC")
    return max_timestamp


if __name__ == "__main__":
    ts = get_curr_oltp_max_timestamp()
    logger.info(f"Max timestamp in OLTP: <ts>{ts}</ts>")
    # Also echo the bare tag pair so `awk -F'<ts>|</ts>' '{print $2}'` works
    # even if loguru formatting shifts.
    print(f"<ts>{ts}</ts>")