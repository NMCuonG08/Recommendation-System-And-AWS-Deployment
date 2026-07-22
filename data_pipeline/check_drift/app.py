"""Evidently AI Data Drift Checker Service — MovieLens Recommendation System.

This service polls change data capture (CDC) records from AWS Kinesis stream ("recsys-cdc"),
accumulates real-time movie rating events, and compares their statistical distribution
against historical reference data (from RDS PostgreSQL "movie_ratings" table or reference parquet)
using Evidently AI DataDriftPreset.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

import boto3
import pandas as pd
from sqlalchemy import create_engine

try:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("check_drift")


def load_configurations() -> tuple[str | None, str | None, str | None, str, int, int]:
    """Loads configuration settings from environment variables."""
    region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1"))
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    stream_name = os.getenv("STREAM_NAME", "recsys-cdc")
    interval = int(os.getenv("INTERVAL", "5"))
    min_messages = int(os.getenv("MIN_MESSAGES", "10"))
    return region, access_key, secret_key, stream_name, interval, min_messages


def initialize_kinesis_client(region: str | None, access_key: str | None, secret_key: str | None):
    """Initializes a boto3 Kinesis client."""
    kwargs: dict[str, Any] = {}
    if region:
        kwargs["region_name"] = region
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("kinesis", **kwargs)


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes data types for MovieLens DataFrame for drift analysis."""
    cols_check = ["userId", "movieId", "rating", "timestamp"]
    df = df.copy()
    for col in cols_check:
        if col not in df.columns:
            df[col] = None

    df["userId"] = pd.to_numeric(df["userId"], errors="coerce").astype("Int64")
    df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce").astype("Int64")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("float64")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s" if df["timestamp"].dtype in ["int64", "float64"] else None, errors="coerce")
    
    if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    return df[cols_check].dropna()


def load_reference_data() -> pd.DataFrame:
    """Loads reference data from PostgreSQL RDS or fallback parquet."""
    use_rds = os.getenv("USE_RDS", "0") == "1"
    if use_rds:
        logger.info("Loading reference data from RDS PostgreSQL...")
        username = os.getenv("RDS_USERNAME", "postgres")
        password = os.getenv("RDS_PASSWORD", "postgres")
        host = os.getenv("RDS_HOST", "localhost")
        port = os.getenv("RDS_PORT", "5432")
        database = os.getenv("RDS_DATABASE", "recsys_oltp")
        table_name = os.getenv("RDS_TABLE", "movie_ratings")
        conn_str = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
        try:
            engine = create_engine(conn_str)
            query = f"SELECT userId, movieId, rating, timestamp FROM {table_name} LIMIT 10000;"
            df_ref = pd.read_sql(query, engine)
            logger.info("Loaded %d reference records from RDS.", len(df_ref))
            return fix_dtypes(df_ref)
        except Exception as err:
            logger.error("Failed to load reference data from RDS: %s. Falling back to synthetic reference.", err)

    ref_parquet = os.getenv("REFERENCE_PARQUET_PATH", "notebooks/data/ml-latest-small/ratings.parquet")
    if os.path.exists(ref_parquet):
        logger.info("Loading reference data from parquet: %s", ref_parquet)
        df_ref = pd.read_parquet(ref_parquet)
        return fix_dtypes(df_ref)

    logger.warning("Reference parquet not found at %s. Creating dummy baseline.", ref_parquet)
    dummy = pd.DataFrame({
        "userId": [1, 2, 3, 4, 5],
        "movieId": [10, 20, 30, 40, 50],
        "rating": [4.0, 3.5, 5.0, 2.0, 4.5],
        "timestamp": [1600000000, 1600000100, 1600000200, 1600000300, 1600000400]
    })
    return fix_dtypes(dummy)


def process_kinesis_stream():
    """Main loop polling Kinesis stream and running Evidently Data Drift checks."""
    region, access_key, secret_key, stream_name, interval, min_messages = load_configurations()
    logger.info("Starting Kinesis Stream Drift Checker for stream: %s", stream_name)

    kinesis = initialize_kinesis_client(region, access_key, secret_key)
    reference_df = load_reference_data()
    logger.info("Reference data shape: %s", reference_df.shape)

    try:
        response = kinesis.describe_stream(StreamName=stream_name)
        shards = response["StreamDescription"]["Shards"]
        shard_id = shards[0]["ShardId"]
        shard_iterator = kinesis.get_shard_iterator(
            StreamName=stream_name,
            ShardId=shard_id,
            ShardIteratorType="LATEST",
        )["ShardIterator"]
    except Exception as err:
        logger.error("Error connecting to Kinesis stream '%s': %s", stream_name, err)
        return

    buffer: list[dict[str, Any]] = []

    while True:
        try:
            records_response = kinesis.get_records(ShardIterator=shard_iterator, Limit=100)
            shard_iterator = records_response.get("NextShardIterator")
            records = records_response.get("Records", [])

            for record in records:
                try:
                    payload = json.loads(record["Data"].decode("utf-8"))
                    data = payload.get("data", payload)
                    if isinstance(data, dict) and "userId" in data:
                        buffer.append(data)
                except Exception as parse_err:
                    logger.debug("Failed parsing record: %s", parse_err)

            if len(buffer) >= min_messages:
                logger.info("Collected %d new streaming records. Running drift check...", len(buffer))
                current_df = fix_dtypes(pd.DataFrame(buffer))
                
                if len(current_df) > 0 and EVIDENTLY_AVAILABLE:
                    report = Report(metrics=[DataDriftPreset()])
                    report.run(reference_data=reference_df, current_data=current_df)
                    report_dict = report.as_dict()
                    drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]
                    logger.info("Evidently Data Drift Analysis Result: dataset_drift=%s", drift_detected)

                    report_out = os.getenv("DRIFT_REPORT_HTML", "/tmp/drift_report.html")
                    report.save_html(report_out)
                    logger.info("Drift report saved to %s", report_out)
                elif not EVIDENTLY_AVAILABLE:
                    logger.warning("Evidently library not installed. Data summary:\n%s", current_df.describe())

                buffer.clear()

        except Exception as loop_err:
            logger.error("Error in drift checking polling loop: %s", loop_err)

        time.sleep(interval)


if __name__ == "__main__":
    process_kinesis_stream()
