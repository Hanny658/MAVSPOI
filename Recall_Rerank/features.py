from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np

from src.agents.utils import (
    clip01,
    exp_distance_score,
    parse_hour,
    parse_price_level,
    purpose_label,
    tokenize,
)
from src.schemas import CandidateScore, UserQueryContext


FEATURE_NAMES: list[str] = [
    "retrieval_score",
    "retrieval_rank_norm",
    "text_similarity",
    "geo_score",
    "popularity_score",
    "stars_norm",
    "review_log_norm",
    "is_open",
    "city_match",
    "state_match",
    "distance_available",
    "distance_decay",
    "within_profile_radius",
    "over_radius_penalty",
    "query_token_overlap",
    "profile_category_overlap",
    "price_match",
    "active_hour_match",
    "urgent_open_match",
    "purpose_match",
]


_PURPOSE_CATEGORY_MAP: dict[str, set[str]] = {
    "study_work": {"coffee", "cafes", "cafe", "coworking", "library", "tea"},
    "social": {"bars", "nightlife", "lounges", "restaurants", "karaoke"},
    "family": {"family", "parks", "playgrounds", "zoos", "museums"},
    "delivery": {"food", "restaurants", "pizza", "sandwiches", "chinese", "mexican"},
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _active_weekend(local_time: str) -> bool:
    local_time = (local_time or "").strip()
    if not local_time:
        return False
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(local_time, fmt).weekday() >= 5
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(local_time.replace("Z", "+00:00")).weekday() >= 5
    except ValueError:
        return False


def _category_overlap(a: set[str], b: set[str]) -> float:
    if not a:
        return 0.0
    return len(a & b) / max(1, len(a))


def _purpose_match(purpose: str, cand_tokens: set[str]) -> float:
    if purpose == "generic":
        return 0.5
    target = _PURPOSE_CATEGORY_MAP.get(purpose, set())
    if not target:
        return 0.5
    return 1.0 if bool(cand_tokens & target) else 0.3


def _distance_features(
    distance_km: float | None,
    radius_km: float,
) -> tuple[float, float, float, float]:
    if distance_km is None:
        return 0.0, 0.5, 0.5, 0.0
    dist = max(0.0, float(distance_km))
    decay = exp_distance_score(dist, radius_km=max(1.0, radius_km))
    within = 1.0 if dist <= radius_km else 0.0
    over = max(0.0, (dist - radius_km) / max(1.0, radius_km))
    return 1.0, decay, within, min(1.0, over)


def build_feature_matrix(
    context: UserQueryContext,
    candidates: list[CandidateScore],
    profile_features: dict[str, Any],
) -> np.ndarray:
    if not candidates:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)

    query_tokens = tokenize(context.query_text)
    hour = parse_hour(context.local_time)
    is_weekend = _active_weekend(context.local_time)
    urgent = bool(query_tokens & {"now", "urgent", "asap", "immediately", "quick", "right"})
    purpose = purpose_label(query_tokens)

    profile_top_categories = {
        str(x).strip().lower()
        for x in profile_features.get("top_categories", [])
        if str(x).strip()
    }
    dominant_price = str(profile_features.get("dominant_price_level", "")).strip()
    active_hours = {
        int(x)
        for x in profile_features.get("active_hours_top3", [])
        if isinstance(x, (int, float, str)) and str(x).isdigit()
    }
    weekend_ratio = clip01(_safe_float(profile_features.get("weekend_ratio", 0.0), 0.0))
    profile_radius = _safe_float(profile_features.get("radius_km_p90", 10.0), 10.0)
    profile_radius = min(30.0, max(2.0, profile_radius))

    max_reviews = max((max(0, c.business.review_count) for c in candidates), default=1)
    max_reviews = max(1, max_reviews)
    query_size = max(1, len(query_tokens))
    city_ref = context.city.strip().lower()
    state_ref = context.state.strip().lower()

    rows: list[list[float]] = []
    n = len(candidates)
    denom = max(1, n - 1)
    for rank_idx, candidate in enumerate(candidates):
        poi = candidate.business
        cand_tokens = tokenize(poi.name + " " + " ".join(poi.categories))
        cand_categories = {
            str(cat).strip().lower() for cat in poi.categories if str(cat).strip()
        }

        stars_norm = clip01(float(poi.stars) / 5.0)
        review_norm = clip01(math.log1p(max(0, poi.review_count)) / math.log1p(max_reviews))
        open_norm = 1.0 if int(poi.is_open) == 1 else 0.0
        city_match = 1.0 if city_ref and poi.city.strip().lower() == city_ref else 0.0
        state_match = 1.0 if state_ref and poi.state.strip().lower() == state_ref else 0.0

        distance_available, distance_decay, within_radius, over_radius = _distance_features(
            candidate.distance_km,
            radius_km=profile_radius,
        )
        query_overlap = len(query_tokens & cand_tokens) / query_size if query_tokens else 0.0
        profile_overlap = _category_overlap(profile_top_categories, cand_categories)

        cand_price = parse_price_level(poi.attributes)
        if dominant_price and cand_price:
            price_match = 1.0 if dominant_price == cand_price else 0.2
        elif dominant_price or cand_price:
            price_match = 0.4
        else:
            price_match = 0.5

        if hour is None or not active_hours:
            active_hour_match = 0.5
        else:
            active_hour_match = 1.0 if hour in active_hours else 0.0

        urgent_open = 1.0 if urgent and open_norm > 0.5 else (0.5 if not urgent else 0.0)
        purpose_match = _purpose_match(purpose, cand_tokens)

        weekend_alignment = 1.0 - abs((1.0 if is_weekend else 0.0) - weekend_ratio)
        active_hour_match = clip01(0.8 * active_hour_match + 0.2 * weekend_alignment)

        row = [
            float(candidate.score),
            1.0 - (float(rank_idx) / float(denom)),
            float(candidate.text_similarity),
            float(candidate.geo_score),
            float(candidate.popularity_score),
            stars_norm,
            review_norm,
            open_norm,
            city_match,
            state_match,
            distance_available,
            distance_decay,
            within_radius,
            over_radius,
            clip01(query_overlap),
            clip01(profile_overlap),
            clip01(price_match),
            active_hour_match,
            urgent_open,
            clip01(purpose_match),
        ]
        rows.append(row)

    return np.asarray(rows, dtype=np.float64)


def feature_tag_hints(feature_row: np.ndarray) -> list[str]:
    if feature_row.size != len(FEATURE_NAMES):
        return ["non_llm_rerank"]
    tags: list[str] = ["non_llm_rerank"]
    if feature_row[2] >= 0.6 or feature_row[14] >= 0.2:
        tags.append("intent_signal")
    if feature_row[11] >= 0.55 or feature_row[12] >= 0.5:
        tags.append("spatial_fit")
    if feature_row[17] >= 0.65:
        tags.append("temporal_fit")
    if feature_row[15] >= 0.25 or feature_row[16] >= 0.6:
        tags.append("profile_fit")
    if feature_row[7] >= 0.9:
        tags.append("open_now")
    return tags[:6]

