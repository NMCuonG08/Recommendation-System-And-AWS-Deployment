"""Locust load test for the API Gateway /recommend endpoint.

Ported from the reference `locustfile.py`, adapted to this port:
  - POST /recommend (not GET /infer) with JSON body {user_id, current_item_id}.
  - MovieLens int ids (not Amazon string user_id).

Run:
    locust -f locustfile.py
Then open http://localhost:8089, set host to the API Gateway URL
(http://<api-gateway-service>/, or http://localhost:8080 for docker-compose).
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task

# A small pool of real MovieLens (ml-latest-small) user + movie ids. The gateway
# resolves candidates from Redis + Feast + Triton; ids absent from caches return
# empty/fallback results but still exercise the full path.
USER_IDS = list(range(1, 611))      # 610 users in ml-latest-small
MOVIE_IDS = list(range(1, 9000))    # ~9.7k movies; sample first 9000


class RecommendUser(HttpUser):
    """Simulate a user requesting recommendations."""

    wait_time = between(0.0001, 0.0002)  # seconds between requests

    @task
    def recommend(self) -> None:
        self.client.post(
            "/recommend",
            json={
                "user_id": random.choice(USER_IDS),
                "current_item_id": random.choice(MOVIE_IDS),
            },
            headers={"Content-Type": "application/json"},
            name="/recommend",
        )
