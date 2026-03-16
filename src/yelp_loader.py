from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.schemas import BusinessPOI


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_categories(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [c.strip() for c in raw.split(",") if c.strip()]


def load_yelp_businesses(
    business_json_path: str,
    max_businesses: int = 50000,
    city_filter: str = "",
) -> list[BusinessPOI]:
    path = Path(business_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Business file not found: {path}")

    city_filter_lower = city_filter.strip().lower()
    businesses: list[BusinessPOI] = []
    for row in _iter_jsonl(path):
        if row.get("is_open", 1) != 1:
            continue
        city_name = str(row.get("city", "")).strip()
        if city_filter_lower and city_name.lower() != city_filter_lower:
            continue
        business = BusinessPOI(
            business_id=str(row.get("business_id", "")).strip(),
            name=str(row.get("name", "")).strip(),
            address=str(row.get("address", "")).strip(),
            city=city_name,
            state=str(row.get("state", "")).strip(),
            latitude=float(row.get("latitude", 0.0)),
            longitude=float(row.get("longitude", 0.0)),
            categories=_parse_categories(row.get("categories")),
            stars=float(row.get("stars", 0.0)),
            review_count=int(row.get("review_count", 0)),
            is_open=int(row.get("is_open", 1)),
            raw_hours=row.get("hours") or {},
            attributes=row.get("attributes") or {},
        )
        if not business.business_id or not business.name:
            continue
        businesses.append(business)
        if len(businesses) >= max_businesses:
            break
    if not businesses:
        raise ValueError("No business records loaded. Check filters/path.")
    return businesses

