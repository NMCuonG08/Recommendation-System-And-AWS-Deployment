"""Feast entity definitions for the MovieLens feature store.

MovieLens ids are integers, so both entities are Int64. The `join_key`
keeps the native MovieLens column names (`userId`, `movieId`) so the
interaction parquets from `notebooks/001` join without renaming.
"""

from feast import Entity, ValueType

user = Entity(
    name="user",
    join_keys=["userId"],
    value_type=ValueType.INT64,
    description="MovieLens user id.",
)

movie = Entity(
    name="movie",
    join_keys=["movieId"],
    value_type=ValueType.INT64,
    description="MovieLens movie id.",
)