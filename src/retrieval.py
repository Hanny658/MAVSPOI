from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.openai_client import OpenAIService
from src.schemas import BusinessPOI, CandidateScore, UserQueryContext
from utils.geo import haversine_km


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class CandidateRetriever:
    def __init__(
        self,
        openai_service: OpenAIService,
        businesses: list[BusinessPOI],
        embed_cache_path: str,
    ) -> None:
        self.openai_service = openai_service
        self.businesses = businesses
        self.embed_cache_path = Path(embed_cache_path)
        self.business_by_id = {b.business_id: b for b in businesses}
        self._embedding_matrix: np.ndarray | None = None
        self._business_order: list[str] = []
        self._id_to_index: dict[str, int] = {}
        self._max_reviews: int = 1

    def _load_cache(self) -> dict[str, list[float]]:
        if not self.embed_cache_path.exists():
            return {}
        cache: dict[str, list[float]] = {}
        with self.embed_cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cache[str(row["business_id"])] = row["embedding"]
        return cache

    def _append_cache_rows(self, rows: list[dict]) -> None:
        self.embed_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.embed_cache_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def prepare_embeddings(self) -> None:
        cache = self._load_cache()
        missing: list[BusinessPOI] = [
            b for b in self.businesses if b.business_id not in cache
        ]

        if missing:
            batch_size = 128
            for i in range(0, len(missing), batch_size):
                batch = missing[i : i + batch_size]
                texts = [b.to_retrieval_text() for b in batch]
                embeds = self.openai_service.embed_texts(texts)
                rows = []
                for b, e in zip(batch, embeds):
                    cache[b.business_id] = e
                    rows.append({"business_id": b.business_id, "embedding": e})
                self._append_cache_rows(rows)

        matrix_rows: list[list[float]] = []
        order: list[str] = []
        for b in self.businesses:
            vec = cache.get(b.business_id)
            if vec is None:
                continue
            matrix_rows.append(vec)
            order.append(b.business_id)
        if not matrix_rows:
            raise ValueError("No embeddings available to build retriever.")
        matrix = np.array(matrix_rows, dtype=np.float32)
        self._embedding_matrix = _normalize_rows(matrix)
        self._business_order = order
        self._id_to_index = {bid: i for i, bid in enumerate(order)}
        self._max_reviews = max((b.review_count for b in self.businesses), default=1)
        self._max_reviews = max(self._max_reviews, 1)

    def _query_vector(self, context: UserQueryContext) -> np.ndarray:
        query_embedding = self.openai_service.embed_texts([self._build_query_text(context)])[0]
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            q_norm = 1.0
        return q / q_norm

    def _score_single(
        self,
        idx: int,
        text_sim: float,
        context: UserQueryContext,
    ) -> CandidateScore:
        business_id = self._business_order[idx]
        poi = self.business_by_id[business_id]
        popularity = 0.5 * (poi.stars / 5.0) + 0.5 * (
            math.log1p(poi.review_count) / math.log1p(self._max_reviews)
        )
        if context.location:
            dist = haversine_km(
                context.location.lat,
                context.location.lon,
                poi.latitude,
                poi.longitude,
            )
            geo_score = math.exp(-dist / 8.0)
            final = 0.65 * float(text_sim) + 0.2 * geo_score + 0.15 * popularity
        else:
            dist = None
            geo_score = 0.0
            final = 0.8 * float(text_sim) + 0.2 * popularity

        if context.city and poi.city.lower() == context.city.lower():
            final += 0.05

        return CandidateScore(
            business=poi,
            score=float(final),
            text_similarity=float(text_sim),
            geo_score=float(geo_score),
            popularity_score=float(popularity),
            distance_km=dist,
        )

    def _build_query_text(self, context: UserQueryContext) -> str:
        city_part = f"{context.city}, {context.state}".strip(", ")
        location_text = (
            f"location=({context.location.lat},{context.location.lon})"
            if context.location
            else "location=unknown"
        )
        return (
            f"User query: {context.query_text}. "
            f"Local time: {context.local_time}. "
            f"City hint: {city_part or 'unknown'}. "
            f"{location_text}. "
            f"Long-term preference note: {context.long_term_notes or 'none'}. "
            f"Recent activity note: {context.recent_activity_notes or 'none'}."
        )

    def retrieve(self, context: UserQueryContext, top_k: int = 80) -> list[CandidateScore]:
        if self._embedding_matrix is None:
            self.prepare_embeddings()
        assert self._embedding_matrix is not None

        q = self._query_vector(context)
        text_sim = self._embedding_matrix @ q
        output = [
            self._score_single(i, float(text_sim[i]), context)
            for i in range(len(self._business_order))
        ]
        output.sort(key=lambda x: x.score, reverse=True)
        return output[:top_k]

    def retrieve_from_pool(
        self,
        context: UserQueryContext,
        candidate_business_ids: list[str],
        top_k: int = 80,
    ) -> list[CandidateScore]:
        if self._embedding_matrix is None:
            self.prepare_embeddings()
        assert self._embedding_matrix is not None

        pool_indices: list[int] = []
        seen: set[str] = set()
        for bid in candidate_business_ids:
            if bid in seen:
                continue
            seen.add(bid)
            idx = self._id_to_index.get(bid)
            if idx is not None:
                pool_indices.append(idx)
        if not pool_indices:
            return self.retrieve(context=context, top_k=top_k)

        q = self._query_vector(context)
        text_sim = self._embedding_matrix @ q
        output = [
            self._score_single(i, float(text_sim[i]), context)
            for i in pool_indices
        ]
        output.sort(key=lambda x: x.score, reverse=True)
        return output[:top_k]
