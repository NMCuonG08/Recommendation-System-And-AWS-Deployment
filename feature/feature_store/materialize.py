"""Apply feature views + materialize offline features into the Feast online store.

Mirrors the reference project's `feast apply && feast materialize` step but as
a runnable module (so it composes with the rest of the local-pro stack and can
be invoked from a container entrypoint / Airflow). Works for both configs:

  - AWS:   `feature_store.yaml`      (DynamoDB online, SQL registry)
  - Local: `feature_store.local.yaml` (Redis online, sqlite registry)

Usage (from feature/feature_store, with Redis/DynamoDB up):
    uv run python materialize.py                                   # local config
    uv run python materialize.py --config feature_store.yaml      # AWS config
    uv run python materialize.py --end 2018-01-01T00:00:00         # custom end ts

The materialize window is [2010-01-01, end_ts]. The lower bound precedes the
oldest MovieLens rating; the upper bound defaults to the latest event_timestamp
in the offline parquet sources so every feature row lands in the online store.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = "feature_store.local.yaml"
WINDOW_START = "2010-01-01T00:00:00"

# Offline parquet sources (must match feature_views.py OUTPUT_DIR).
MOVIE_PARQUET = REPO_DIR.parent / "output" / "movie_rating_stats.parquet"
USER_PARQUET = REPO_DIR.parent / "output" / "user_rating_stats.parquet"


def _latest_event_timestamp() -> str:
    """Return ISO ts of the newest event_timestamp across the offline sources."""
    latest = pd.Timestamp.min.tz_localize("UTC")
    for path in (MOVIE_PARQUET, USER_PARQUET):
        if not path.is_file():
            continue
        df = pd.read_parquet(path, columns=["event_timestamp"])
        ts = pd.to_datetime(df["event_timestamp"], utc=True).max()
        if ts > latest:
            latest = ts
    if latest == pd.Timestamp.min.tz_localize("UTC"):
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return latest.tz_convert("UTC").tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feast apply + materialize (MovieLens).")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Feast config yaml (default: feature_store.local.yaml).")
    parser.add_argument("--start", default=WINDOW_START,
                        help="Materialize window start (default: 2010-01-01T00:00:00).")
    parser.add_argument("--end", default=None,
                        help="Materialize window end (default: latest event_timestamp).")
    parser.add_argument("--no-apply", action="store_true",
                        help="Skip `feast apply` (registry already up to date).")
    args = parser.parse_args()

    config_path = REPO_DIR / args.config
    if not config_path.is_file():
        logger.error("Feast config not found: %s", config_path)
        sys.exit(1)

    # Activate the chosen config as feature_store.yaml (Feast reads this name).
    active_yaml = REPO_DIR / "feature_store.yaml"
    if config_path != active_yaml:
        try:
            logger.info("Activating config %s -> feature_store.yaml", args.config)
            active_yaml.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            logger.info("Config already active or read-only filesystem; proceeding with active config.")

    from feast import FeatureStore

    store = FeatureStore(repo_path=str(REPO_DIR), fs_yaml_file="feature_store.yaml")

    if not args.no_apply:
        logger.info("Applying feature definitions...")
        store.apply(store.list_feature_views())
        logger.info("Feast apply complete.")

    end_ts = args.end or _latest_event_timestamp()
    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_ts).replace(tzinfo=timezone.utc) if isinstance(end_ts, str) else end_ts
    logger.info("Materializing online features [%s, %s]...", start_dt, end_dt)
    store.materialize(start_date=start_dt, end_date=end_dt)
    logger.info("Materialize complete. Online store populated.")


if __name__ == "__main__":
    main()