"""Model training phases for the MovieLens recsys pipeline.

Mirrors the `feature/` phase-dir convention: each model family lives in its
own subpackage (`item2vec/`, later `ranking_sequence/`), with a numbered
notebook driving it and outputs under `models/output/<model>/`.
"""