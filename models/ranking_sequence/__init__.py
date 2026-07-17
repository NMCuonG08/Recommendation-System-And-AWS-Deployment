"""GRU sequence ranker — consumes the Item2Vec champion embeddings.

Ported from the reference `src/model_ranking_sequence/`. Trains a GRU ranker
over user item sequences with frozen Item2Vec item embeddings, registers the
champion in MLflow (`{run_name}_sequence_rating`).
"""