"""Time-difference bucketization — copy of feature/features/timestamp_bucket.py.

Copied into the Lambda image (self-contained: the image does not import the
repo's ``feature`` package). Must stay byte-identical with the offline pipeline
so online and offline sequence buckets agree.
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
    """Bucket for ``ts`` relative to ``current_ts`` (defaults to now)."""
    if current_ts is None:
        current_ts = int(time.time())
    return bucketize_seconds_diff(current_ts - ts)