"""Data preparation helpers for the HQ Trivia sample dataset.

Schema of the Kaggle dataset `hqinsiders/hq-trivia-sample` is not known up
front, so these utilities are discovery-based: they auto-detect file formats
and timestamp encoding rather than assuming fixed column names.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

# File extensions we know how to load as a tabular frame.
_TABLE_EXTS = {".csv", ".tsv", ".parquet", ".json", ".jsonl", ".txt"}


def discover_dataset_files(dataset_path: str) -> pd.DataFrame:
    """Walk a kagglehub dataset directory and return a manifest of files.

    Parameters
    ----------
    dataset_path:
        Path returned by ``kagglehub.dataset_download``.

    Returns
    -------
    DataFrame with columns: relative_path, abspath, ext, size_bytes, size_mb.
    """
    rows = []
    root = Path(dataset_path)
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            abspath = Path(dirpath) / name
            rel = abspath.relative_to(root)
            size = abspath.stat().st_size
            rows.append(
                {
                    "relative_path": str(rel),
                    "abspath": str(abspath),
                    "ext": abspath.suffix.lower(),
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 4),
                }
            )
    manifest = pd.DataFrame(rows).sort_values("size_bytes", ascending=False)
    return manifest.reset_index(drop=True)


def load_table(path: str, ext: Optional[str] = None) -> pd.DataFrame:
    """Load a single file into a DataFrame based on its extension.

    JSON files may be either a list of records, a single object, or JSONL.
    Nested fields are flattened with ``json_normalize`` for record lists.
    """
    ext = (ext or Path(path).suffix).lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            # Either a single record or a dict-of-columns.
            if any(isinstance(v, list) for v in payload.values()):
                return pd.json_normalize(payload)
            return pd.DataFrame([payload])
        raise ValueError(f"Unsupported JSON root type in {path}: {type(payload)}")
    if ext == ".txt":
        # Best-effort: try CSV first, fall back to whitespace-delimited.
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")
    raise ValueError(f"Unsupported file extension {ext!r} for {path}")


def load_all_tables(dataset_path: str) -> Dict[str, pd.DataFrame]:
    """Load every table-like file in a dataset dir into a dict keyed by stem.

    Files whose extension is not in :data:`_TABLE_EXTS` are skipped.
    """
    manifest = discover_dataset_files(dataset_path)
    tables: Dict[str, pd.DataFrame] = {}
    for _, row in manifest.iterrows():
        if row["ext"] not in _TABLE_EXTS:
            continue
        stem = Path(row["relative_path"]).stem
        try:
            tables[stem] = load_table(row["abspath"], row["ext"])
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[load_all_tables] skip {row['relative_path']}: {exc}")
    return tables


def _guess_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    """Return the first column whose name looks like a timestamp."""
    candidates = {"timestamp", "time", "ts", "created_at", "created", "date", "datetime"}
    for col in df.columns:
        if col.lower() in candidates:
            return col
    return None


def parse_dt(df: pd.DataFrame, cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Convert timestamp columns to datetime, auto-detecting the unit.

    Heuristic: if the numeric max is large (>1e12) it is in milliseconds,
    otherwise seconds; non-numeric values are parsed as ISO strings.
    """
    if cols is None:
        guessed = _guess_timestamp_col(df)
        cols = [guessed] if guessed else []
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        if series.notna().any() and (series.max() > 1e12):
            out[col] = pd.to_datetime(series, unit="ms", errors="coerce")
        elif series.notna().any():
            out[col] = pd.to_datetime(series, unit="s", errors="coerce")
        else:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def basic_report(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    """Print + return a per-column report: dtype, nulls, unique, sample."""
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "n_null": int(series.isna().sum()),
                "pct_null": round(series.isna().mean() * 100, 2),
                "n_unique": int(series.nunique(dropna=True)),
                "sample": str(series.dropna().iloc[0])[:60] if series.notna().any() else "",
            }
        )
    report = pd.DataFrame(rows)
    print(f"\n=== {name} | shape={df.shape} ===")
    print(report.to_string(index=False))
    return report


def calculate_sparsity(
    df: pd.DataFrame, user_col: str, item_col: str
) -> float:
    """Fraction of empty cells in the user x item matrix."""
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    if n_users == 0 or n_items == 0:
        return float("nan")
    return 1 - df.shape[0] / (n_users * n_items)


def filter_min_interactions(
    df: pd.DataFrame, user_col: str, item_col: str,
    min_user: int = 5, min_item: int = 5,
) -> pd.DataFrame:
    """Iteratively drop users/items below an interaction threshold (k-core)."""
    out = df.copy()
    for _ in range(20):  # bounded iteration until stable
        before = len(out)
        user_counts = out[user_col].value_counts()
        item_counts = out[item_col].value_counts()
        keep_users = user_counts[user_counts >= min_user].index
        keep_items = item_counts[item_counts >= min_item].index
        out = out[out[user_col].isin(keep_users) & out[item_col].isin(keep_items)]
        if len(out) == before:
            break
    return out