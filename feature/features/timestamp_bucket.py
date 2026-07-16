"""Time-difference bucketization for user rating sequences.

Ported verbatim from the reference project's
`src/feature_engineer/features/timestamp_bucket.py`. Maps a lag in seconds
between a past rating and the current event to one of 10 discrete buckets,
used as a positional/temporal encoding for the sequence ranking model.
"""

from __future__ import annotations

import time

SEC_MIN = 60
SEC_HOUR = 60 * 60
SEC_DAY = 60 * 60 * 24
SEC_WEEK = 60 * 60 * 24 * 7
SEC_MONTH = 60 * 60 * 24 * 30
SEC_YEAR = 60 * 60 * 24 * 365


def bucketize_seconds_diff(seconds: int) -> int:
    """Convert a time difference in seconds to a discrete bucket index (0-9)."""
    if seconds < SEC_MIN * 10:        # 10 minutes
        return 0
    if seconds < SEC_HOUR:            # 1 hour
        return 1
    if seconds < SEC_DAY:             # 1 day
        return 2
    if seconds < SEC_WEEK:            # 1 week
        return 3
    if seconds < SEC_MONTH:           # 1 month
        return 4
    if seconds < SEC_YEAR:            # 1 year
        return 5
    if seconds < SEC_YEAR * 3:        # 3 years
        return 6
    if seconds < SEC_YEAR * 5:        # 5 years
        return 7
    if seconds < SEC_YEAR * 10:       # 10 years
        return 8
    return 9


def from_ts_to_bucket(ts: int, current_ts: int | None = None) -> int:
    """Bucket for `ts` relative to `current_ts` (defaults to now)."""
    if current_ts is None:
        current_ts = int(time.time())
    return bucketize_seconds_diff(current_ts - ts)


def calc_sequence_timestamp_bucket(row: dict) -> list:
    """Convert a row's `item_sequence_ts` list to bucket indices.

    Padding entries (-1) are preserved as -1.
    """
    ts = row["timestamp_unix"]
    output = []
    for x in row["item_sequence_ts"]:
        x_i = int(x)
        if x_i == -1:
            output.append(x_i)        # keep padding
        else:
            output.append(from_ts_to_bucket(x_i, ts))
    return output