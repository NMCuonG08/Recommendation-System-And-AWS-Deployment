"""sklearn preprocessing pipeline steps for MovieLens features.

Ported from the reference project's `src/feature_engineer/features/tfm.py`,
trimmed to the columns MovieLens actually has:

- `title`      -> TF-IDF (MovieLens has movie titles, no `description`)
- `genres`     -> multi-label count vectorizer (pipe-delimited string)
- rating aggregates -> StandardScaler

E-commerce-only pipelines (`price`, `main_category`, `description`) are
dropped because MovieLens has no price / category / description columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def reshape_2d_to_1d(X):
    """Flatten a (n,1) column to (n,) for text vectorizers."""
    return np.array(X).reshape(-1)


def todense(X):
    """Convert a sparse matrix to a dense ndarray."""
    return np.asarray(X.todense())


def flatten_string_array_col(X):
    """Join a Series of string-lists into a single newline-separated string."""
    assert isinstance(X, pd.Series)
    output = X.fillna("").str.join("\n")
    assert X.shape[0] == output.shape[0]
    return output.values


def tokenizer(s: str):
    """Tokenize by newline (used for pipe-split genres)."""
    return s.split("\n")


def title_pipeline_steps():
    """TF-IDF pipeline for movie `title` text."""
    steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="")),
        ("reshape", FunctionTransformer(reshape_2d_to_1d, validate=False)),
        ("tfidf", TfidfVectorizer(min_df=5, max_features=1000, ngram_range=(1, 2))),
        ("todense", FunctionTransformer(todense, validate=False)),
    ]
    return steps


def genres_pipeline_steps():
    """Count-vectorizer pipeline for pipe-delimited `genres`.

    Genres arrive as `"Adventure|Animation|Comedy"`; we replace `|` with
    newline so the newline tokenizer yields one token per genre.
    """
    def replace_pipe(X):
        return X.fillna("").astype(str).str.replace("|", "\n", regex=False).values

    steps = [
        ("replace_pipe", FunctionTransformer(replace_pipe, validate=False)),
        ("count_vect", CountVectorizer(tokenizer=tokenizer, token_pattern=None)),
        ("todense", FunctionTransformer(todense, validate=False)),
    ]
    return steps


def rating_agg_pipeline_steps():
    """StandardScaler pipeline for numeric rating-aggregate features."""
    steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("normalize", StandardScaler()),
    ]
    return steps