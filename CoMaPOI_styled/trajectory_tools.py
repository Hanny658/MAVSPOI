from __future__ import annotations

from collections import Counter
from typing import Any

from src.agents.utils import parse_hour, tokenize
from src.schemas import CandidateScore, UserQueryContext


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(len(sorted_vals) - 1, lo + 1)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _purpose_heuristic(query_tokens: set[str]) -> str:
    if query_tokens & {"study", "work", "meeting", "quiet", "wifi", "laptop"}:
        return "study_work"
    if query_tokens & {"friends", "date", "party", "hangout", "social"}:
        return "social"
    if query_tokens & {"family", "kids", "children"}:
        return "family"
    if query_tokens & {"delivery", "takeout", "pickup"}:
        return "delivery"
    return "generic"


def build_profiler_tools(
    context: UserQueryContext,
    profile_features: dict[str, Any],
    initial_candidates: list[CandidateScore],
) -> dict[str, Any]:
    top_categories = [
        str(x).strip() for x in profile_features.get("top_categories", []) if str(x).strip()
    ]
    category_counter: Counter[str] = Counter()
    city_counter: Counter[str] = Counter()
    open_count = 0
    distances: list[float] = []
    for candidate in initial_candidates[:50]:
        poi = candidate.business
        city_counter[f"{poi.city},{poi.state}"] += 1
        for cat in poi.categories[:3]:
            cat_text = str(cat).strip().lower()
            if cat_text:
                category_counter[cat_text] += 1
        if int(poi.is_open) == 1:
            open_count += 1
        if candidate.distance_km is not None:
            distances.append(float(candidate.distance_km))

    query_tokens = tokenize(context.query_text)
    hour = parse_hour(context.local_time)
    tools = {
        "tool_freq": {
            "interaction_evidence_count": int(profile_features.get("interaction_evidence_count", 0) or 0),
            "support_level": str(profile_features.get("support_level", "unknown")),
            "retrieval_pool_size": len(initial_candidates),
        },
        "tool_cat": {
            "profile_top_categories": top_categories[:8],
            "retrieval_top_categories": [
                {"category": cat, "count": cnt}
                for cat, cnt in category_counter.most_common(8)
            ],
            "profile_category_overlap_ratio": (
                len(set(x.lower() for x in top_categories) & set(category_counter.keys()))
                / max(1, len(set(x.lower() for x in top_categories)))
                if top_categories
                else 0.0
            ),
        },
        "tool_time": {
            "query_hour": hour,
            "active_hours_top3": profile_features.get("active_hours_top3", []),
            "weekend_ratio": profile_features.get("weekend_ratio", 0.0),
            "query_purpose": _purpose_heuristic(query_tokens),
        },
        "tool_loc": {
            "query_city": context.city,
            "query_state": context.state,
            "retrieval_city_distribution_top5": [
                {"city_state": city, "count": cnt} for city, cnt in city_counter.most_common(5)
            ],
            "distance_km_p50": _quantile(distances, 0.5),
            "distance_km_p90": _quantile(distances, 0.9),
            "profile_radius_km_p90": profile_features.get("radius_km_p90"),
        },
        "tool_poi": {
            "retrieval_open_ratio": round(open_count / max(1, len(initial_candidates[:50])), 6),
            "candidate_snapshot": [c.to_compact_dict() for c in initial_candidates[:10]],
        },
    }
    return tools

