from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def semantic_norm(similarity: float) -> float:
    return clip01((float(similarity) + 1.0) / 2.0)


def parse_hour(local_time: str) -> int | None:
    local_time = (local_time or "").strip()
    if not local_time:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(local_time, fmt).hour
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(local_time.replace("Z", "+00:00")).hour
    except ValueError:
        return None


def parse_bool_text(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "'true'"}


def parse_price_level(attributes: dict[str, Any]) -> str:
    if not isinstance(attributes, dict):
        return ""
    raw = attributes.get("RestaurantsPriceRange2")
    if raw is None:
        return ""
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return digits if digits in {"1", "2", "3", "4"} else ""


def purpose_label(query_tokens: set[str]) -> str:
    if query_tokens & {
        "study",
        "work",
        "meeting",
        "quiet",
        "laptop",
        "wifi",
        "office",
    }:
        return "study_work"
    if query_tokens & {"friends", "date", "party", "hangout", "group", "social"}:
        return "social"
    if query_tokens & {"family", "kids", "children"}:
        return "family"
    if query_tokens & {"delivery", "takeout", "online", "pickup"}:
        return "delivery"
    return "generic"


def exp_distance_score(distance_km: float, radius_km: float) -> float:
    radius_km = max(1.0, radius_km)
    return clip01(math.exp(-float(distance_km) / radius_km))
