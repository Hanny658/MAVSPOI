from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.openai_client import OpenAIService
from src.schemas import BusinessPOI, CandidateScore, UserQueryContext
from utils.geo import haversine_km

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


class CandidateRetriever:
    def __init__(
        self,
        openai_service: OpenAIService,
        businesses: list[BusinessPOI],
        embed_cache_path: str,
    ) -> None:
        self.openai_service = openai_service
        self.businesses = businesses
        self.business_by_id = {b.business_id: b for b in businesses}

        raw_cache = Path(embed_cache_path)
        self.index_path = self._resolve_index_path(raw_cache)
        self.meta_path = self._resolve_meta_path(self.index_path)

        self._index: Any = None
        self._business_order: list[str] = []
        self._id_to_index: dict[str, int] = {}
        self._max_reviews: int = 1

    @staticmethod
    def _resolve_index_path(path: Path) -> Path:
        if path.suffix.lower() == ".faiss":
            return path
        return path.with_suffix(".faiss")

    @staticmethod
    def _resolve_meta_path(index_path: Path) -> Path:
        return index_path.with_suffix(index_path.suffix + ".ids.json")

    @staticmethod
    def _require_faiss() -> None:
        if faiss is None:  # pragma: no cover
            raise RuntimeError(
                "FAISS is required but not installed. Install dependency `faiss-cpu`."
            ) from _FAISS_IMPORT_ERROR

    def _load_index_and_ids(self) -> tuple[Any, list[str]]:
        if not self.index_path.exists() or not self.meta_path.exists():
            return None, []
        self._require_faiss()

        index = faiss.read_index(str(self.index_path))
        with self.meta_path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        ids = row.get("business_ids", [])
        if not isinstance(ids, list):
            raise ValueError(f"Invalid ID metadata file: {self.meta_path}")
        ids = [str(x).strip() for x in ids if str(x).strip()]
        if int(index.ntotal) != len(ids):
            raise ValueError(
                f"FAISS index size ({index.ntotal}) does not match ID count ({len(ids)})."
            )
        return index, ids

    def _save_index_and_ids(self) -> None:
        self._require_faiss()
        if self._index is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump({"business_ids": self._business_order}, f, ensure_ascii=False)

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
        text_similarity: float,
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
            final = 0.65 * float(text_similarity) + 0.2 * geo_score + 0.15 * popularity
        else:
            dist = None
            geo_score = 0.0
            final = 0.8 * float(text_similarity) + 0.2 * popularity

        if context.city and poi.city.lower() == context.city.lower():
            final += 0.05

        return CandidateScore(
            business=poi,
            score=float(final),
            text_similarity=float(text_similarity),
            geo_score=float(geo_score),
            popularity_score=float(popularity),
            distance_km=dist,
        )

    def prepare_embeddings(self) -> None:
        self._require_faiss()

        index, ids = self._load_index_and_ids()
        self._index = index
        self._business_order = ids
        existing_ids = set(ids)

        missing: list[BusinessPOI] = [
            b for b in self.businesses if b.business_id not in existing_ids
        ]

        if missing:
            batch_size = 128
            for i in range(0, len(missing), batch_size):
                batch = missing[i : i + batch_size]
                texts = [b.to_retrieval_text() for b in batch]
                embeds = self.openai_service.embed_texts(texts)
                vec = np.array(embeds, dtype=np.float32)
                vec = _normalize_rows(vec)
                if self._index is None:
                    self._index = faiss.IndexFlatIP(vec.shape[1])
                if int(self._index.d) != int(vec.shape[1]):
                    raise ValueError(
                        f"Embedding dimension mismatch: index={self._index.d}, batch={vec.shape[1]}"
                    )
                self._index.add(vec)
                self._business_order.extend([b.business_id for b in batch])

        if self._index is None:
            raise ValueError("No embeddings available to build FAISS retriever.")

        self._id_to_index = {bid: idx for idx, bid in enumerate(self._business_order)}
        self._max_reviews = max((b.review_count for b in self.businesses), default=1)
        self._max_reviews = max(self._max_reviews, 1)

        if int(self._index.ntotal) != len(self._business_order):
            raise ValueError(
                f"FAISS index total ({self._index.ntotal}) != IDs ({len(self._business_order)})."
            )
        self._save_index_and_ids()

    def retrieve(self, context: UserQueryContext, top_k: int = 80) -> list[CandidateScore]:
        if self._index is None:
            self.prepare_embeddings()
        assert self._index is not None

        q = self._query_vector(context)
        # Retrieve a wider pool by semantic similarity, then apply final scoring.
        preselect_k = min(
            len(self._business_order),
            max(top_k, top_k * 20, 200),
        )
        sims, idxs = self._index.search(q.reshape(1, -1), preselect_k)
        output: list[CandidateScore] = []
        for pos, sim in zip(idxs[0], sims[0]):
            if pos < 0:
                continue
            output.append(self._score_single(int(pos), float(sim), context))
        output.sort(key=lambda x: x.score, reverse=True)
        return output[:top_k]

    def retrieve_from_pool(
        self,
        context: UserQueryContext,
        candidate_business_ids: list[str],
        top_k: int = 80,
    ) -> list[CandidateScore]:
        if self._index is None:
            self.prepare_embeddings()
        assert self._index is not None

        query_ids = _dedup_keep_order([str(x).strip() for x in candidate_business_ids if str(x).strip()])
        if not query_ids:
            return self.retrieve(context=context, top_k=top_k)

        q = self._query_vector(context)
        output: list[CandidateScore] = []
        for bid in query_ids:
            pos = self._id_to_index.get(bid)
            if pos is None:
                continue
            vec = self._index.reconstruct(int(pos))
            sim = float(np.dot(q, vec))
            output.append(self._score_single(int(pos), sim, context))
        if not output:
            return self.retrieve(context=context, top_k=top_k)
        output.sort(key=lambda x: x.score, reverse=True)
        return output[:top_k]

