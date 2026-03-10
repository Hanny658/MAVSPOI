from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.geo import haversine_km


@dataclass
class BusinessInfo:
    business_id: str
    latitude: float | None
    longitude: float | None
    categories: list[str]
    stars: float
    review_count: int
    price_level: int | None


@dataclass
class CheckinInfo:
    total: int = 0
    hour_hist: list[int] = field(default_factory=lambda: [0] * 24)
    weekday_hist: list[int] = field(default_factory=lambda: [0] * 7)


@dataclass
class UserAccumulator:
    # Evidence counters
    review_count: int = 0
    tip_count: int = 0
    interaction_count: int = 0

    # User feedback from reviews/tips
    useful_sum: int = 0
    funny_sum: int = 0
    cool_sum: int = 0
    tip_compliment_sum: int = 0

    # Rating behavior (reviews only)
    rating_sum: float = 0.0
    rating_sq_sum: float = 0.0
    positive_count: int = 0
    negative_count: int = 0

    # Temporal behavior from interaction timestamps
    hour_hist: list[int] = field(default_factory=lambda: [0] * 24)
    weekday_hist: list[int] = field(default_factory=lambda: [0] * 7)
    weekend_count: int = 0

    # Preference signals from visited businesses
    business_ids: set[str] = field(default_factory=set)
    category_counter: Counter[str] = field(default_factory=Counter)
    price_counter: Counter[str] = field(default_factory=Counter)
    business_stars_sum: float = 0.0
    business_review_log_sum: float = 0.0
    business_quality_count: int = 0

    # Aggregated checkin context of visited businesses
    business_checkin_total_sum: int = 0
    business_checkin_count: int = 0
    checkin_hour_hist: list[int] = field(default_factory=lambda: [0] * 24)
    checkin_weekday_hist: list[int] = field(default_factory=lambda: [0] * 7)

    # Spatial points to estimate mobility footprint
    geo_points: list[tuple[float, float]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-user profiles from city-level Yelp subset files. "
            "Input files are read as data/<prefix>-*.jsonl first, then fallback to .json."
        )
    )
    parser.add_argument(
        "--data-prefix",
        default="yelp-indianapolis",
        help="Dataset prefix in data directory (default: yelp-indianapolis).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory path (default: data).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200000,
        help="Progress log frequency by processed lines (default: 200000).",
    )
    return parser.parse_args()


def _resolve_path(data_dir: Path, prefix: str, suffix: str) -> Path:
    # Prefer .jsonl and fallback to .json for compatibility.
    for ext in (".jsonl", ".json"):
        path = data_dir / f"{prefix}-{suffix}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Cannot find file for suffix '{suffix}'. "
        f"Expected {prefix}-{suffix}.jsonl or .json under {data_dir}."
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_price_level(attributes: Any) -> int | None:
    if not isinstance(attributes, dict):
        return None
    raw = attributes.get("RestaurantsPriceRange2")
    if raw is None:
        return None
    # Yelp values may appear as "2", "'2'", or similar variants.
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    if not digits:
        return None
    try:
        value = int(digits)
    except ValueError:
        return None
    if 1 <= value <= 4:
        return value
    return None


def _parse_categories(raw: Any) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _safe_weekday(date_text: str) -> int | None:
    # Input date is expected as YYYY-MM-DD.
    try:
        return date.fromisoformat(date_text).weekday()
    except ValueError:
        return None


def _update_temporal(acc: UserAccumulator, date_time_text: str) -> None:
    if len(date_time_text) < 13:
        return
    hour = _parse_int(date_time_text[11:13], -1)
    if 0 <= hour <= 23:
        acc.hour_hist[hour] += 1
    weekday = _safe_weekday(date_time_text[:10])
    if weekday is not None:
        acc.weekday_hist[weekday] += 1
        if weekday >= 5:
            acc.weekend_count += 1


def _add_business_features(
    acc: UserAccumulator,
    business: BusinessInfo,
    checkin: CheckinInfo | None,
) -> None:
    acc.business_ids.add(business.business_id)
    for category in business.categories:
        acc.category_counter[category] += 1
    if business.price_level is not None:
        acc.price_counter[str(business.price_level)] += 1
    acc.business_stars_sum += business.stars
    acc.business_review_log_sum += math.log1p(max(0, business.review_count))
    acc.business_quality_count += 1

    if business.latitude is not None and business.longitude is not None:
        acc.geo_points.append((business.latitude, business.longitude))

    if checkin and checkin.total > 0:
        acc.business_checkin_total_sum += checkin.total
        acc.business_checkin_count += 1
        for i in range(24):
            acc.checkin_hour_hist[i] += checkin.hour_hist[i]
        for i in range(7):
            acc.checkin_weekday_hist[i] += checkin.weekday_hist[i]


def _normalize_hist(hist: list[int]) -> list[float]:
    total = sum(hist)
    if total <= 0:
        return [0.0] * len(hist)
    return [round(x / total, 6) for x in hist]


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    weight = pos - lo
    return sorted_vals[lo] * (1 - weight) + sorted_vals[hi] * weight


def _top_indices(hist: list[int], k: int) -> list[int]:
    ranked = sorted(range(len(hist)), key=lambda i: hist[i], reverse=True)
    return ranked[:k]


def _support_level(interaction_count: int) -> str:
    # Keep explicit support buckets for fair zero-shot/few-shot evaluation splits.
    if interaction_count <= 0:
        return "zero_shot"
    if interaction_count <= 10:
        return "few_shot"
    return "warm"


def _friend_count(friends_text: str) -> int:
    if not friends_text or friends_text == "None":
        return 0
    return len([x for x in friends_text.split(",") if x.strip()])


def _elite_count(elite_text: str) -> int:
    if not elite_text:
        return 0
    return len([x for x in elite_text.split(",") if x.strip()])


def load_business_map(path: Path) -> dict[str, BusinessInfo]:
    business_map: dict[str, BusinessInfo] = {}
    for row in _iter_jsonl(path):
        bid = str(row.get("business_id", "")).strip()
        if not bid:
            continue
        lat = row.get("latitude")
        lon = row.get("longitude")
        latitude = _parse_float(lat, 0.0) if lat is not None else None
        longitude = _parse_float(lon, 0.0) if lon is not None else None
        business_map[bid] = BusinessInfo(
            business_id=bid,
            latitude=latitude,
            longitude=longitude,
            categories=_parse_categories(row.get("categories")),
            stars=_parse_float(row.get("stars", 0.0)),
            review_count=_parse_int(row.get("review_count", 0)),
            price_level=_parse_price_level(row.get("attributes")),
        )
    return business_map


def load_checkin_map(path: Path, valid_business_ids: set[str]) -> dict[str, CheckinInfo]:
    checkin_map: dict[str, CheckinInfo] = {}
    for row in _iter_jsonl(path):
        bid = str(row.get("business_id", "")).strip()
        if not bid or bid not in valid_business_ids:
            continue
        raw_dates = str(row.get("date", "")).strip()
        if not raw_dates:
            continue
        info = CheckinInfo()
        for token in raw_dates.split(","):
            dt = token.strip()
            if len(dt) < 13:
                continue
            hour = _parse_int(dt[11:13], -1)
            if 0 <= hour <= 23:
                info.hour_hist[hour] += 1
            weekday = _safe_weekday(dt[:10])
            if weekday is not None:
                info.weekday_hist[weekday] += 1
            info.total += 1
        checkin_map[bid] = info
    return checkin_map


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    prefix = args.data_prefix.strip()
    if not prefix:
        raise ValueError("--data-prefix cannot be empty")

    business_path = _resolve_path(data_dir, prefix, "business")
    review_path = _resolve_path(data_dir, prefix, "review")
    tip_path = _resolve_path(data_dir, prefix, "tip")
    user_path = _resolve_path(data_dir, prefix, "user")
    checkin_path = _resolve_path(data_dir, prefix, "checkin")
    output_path = data_dir / f"{prefix}-profile.jsonl"

    print(f"[1/6] Loading business map from {business_path}")
    business_map = load_business_map(business_path)
    print(f"  loaded businesses: {len(business_map):,}")

    print(f"[2/6] Loading checkin map from {checkin_path}")
    checkin_map = load_checkin_map(checkin_path, set(business_map.keys()))
    print(f"  loaded checkin rows: {len(checkin_map):,}")

    print(f"[3/6] Aggregating review interactions from {review_path}")
    user_acc: dict[str, UserAccumulator] = defaultdict(UserAccumulator)
    for i, row in enumerate(_iter_jsonl(review_path), 1):
        uid = str(row.get("user_id", "")).strip()
        bid = str(row.get("business_id", "")).strip()
        if not uid:
            continue
        acc = user_acc[uid]
        acc.review_count += 1
        acc.interaction_count += 1
        acc.useful_sum += _parse_int(row.get("useful", 0))
        acc.funny_sum += _parse_int(row.get("funny", 0))
        acc.cool_sum += _parse_int(row.get("cool", 0))

        stars = _parse_float(row.get("stars", 0.0))
        if stars > 0:
            acc.rating_sum += stars
            acc.rating_sq_sum += stars * stars
            if stars >= 4.0:
                acc.positive_count += 1
            if stars <= 2.0:
                acc.negative_count += 1

        _update_temporal(acc, str(row.get("date", "")))

        business = business_map.get(bid)
        if business:
            _add_business_features(acc, business, checkin_map.get(bid))

        if i % args.progress_every == 0:
            print(f"  review processed: {i:,}")

    print(f"  users after review pass: {len(user_acc):,}")

    print(f"[4/6] Aggregating tip interactions from {tip_path}")
    for i, row in enumerate(_iter_jsonl(tip_path), 1):
        uid = str(row.get("user_id", "")).strip()
        bid = str(row.get("business_id", "")).strip()
        if not uid:
            continue
        acc = user_acc[uid]
        acc.tip_count += 1
        acc.interaction_count += 1
        acc.tip_compliment_sum += _parse_int(row.get("compliment_count", 0))
        _update_temporal(acc, str(row.get("date", "")))

        business = business_map.get(bid)
        if business:
            _add_business_features(acc, business, checkin_map.get(bid))

        if i % args.progress_every == 0:
            print(f"  tip processed: {i:,}")

    print(f"  users after tip pass: {len(user_acc):,}")

    print(f"[5/6] Loading user static attributes from {user_path}")
    user_static_map: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(_iter_jsonl(user_path), 1):
        uid = str(row.get("user_id", "")).strip()
        if not uid:
            continue
        user_static_map[uid] = {
            "name": str(row.get("name", "")),
            "yelping_since": str(row.get("yelping_since", "")),
            "user_review_count": _parse_int(row.get("review_count", 0)),
            "user_average_stars": _parse_float(row.get("average_stars", 0.0)),
            "fans": _parse_int(row.get("fans", 0)),
            "friend_count": _friend_count(str(row.get("friends", ""))),
            "elite_years_count": _elite_count(str(row.get("elite", ""))),
            "compliment_total": sum(
                _parse_int(row.get(key, 0))
                for key in [
                    "compliment_hot",
                    "compliment_more",
                    "compliment_profile",
                    "compliment_cute",
                    "compliment_list",
                    "compliment_note",
                    "compliment_plain",
                    "compliment_cool",
                    "compliment_funny",
                    "compliment_writer",
                    "compliment_photos",
                ]
            ),
        }
        if i % args.progress_every == 0:
            print(f"  user processed: {i:,}")

    all_user_ids = sorted(set(user_static_map.keys()) | set(user_acc.keys()))
    print(f"[6/6] Building profile rows for users: {len(all_user_ids):,}")
    with output_path.open("w", encoding="utf-8") as fw:
        for uid in all_user_ids:
            acc = user_acc.get(uid, UserAccumulator())
            static = user_static_map.get(
                uid,
                {
                    "name": "",
                    "yelping_since": "",
                    "user_review_count": 0,
                    "user_average_stars": 0.0,
                    "fans": 0,
                    "friend_count": 0,
                    "elite_years_count": 0,
                    "compliment_total": 0,
                },
            )

            rated = acc.review_count
            avg_review_stars = (acc.rating_sum / rated) if rated > 0 else 0.0
            variance = (acc.rating_sq_sum / rated - avg_review_stars**2) if rated > 0 else 0.0
            std_review_stars = math.sqrt(max(0.0, variance))
            positive_ratio = (acc.positive_count / rated) if rated > 0 else 0.0
            negative_ratio = (acc.negative_count / rated) if rated > 0 else 0.0

            interaction_total = acc.interaction_count
            weekend_ratio = (acc.weekend_count / interaction_total) if interaction_total > 0 else 0.0

            lat_center = None
            lon_center = None
            avg_radius = None
            p50_radius = None
            p90_radius = None
            if acc.geo_points:
                lat_center = sum(lat for lat, _ in acc.geo_points) / len(acc.geo_points)
                lon_center = sum(lon for _, lon in acc.geo_points) / len(acc.geo_points)
                distances = [
                    haversine_km(lat_center, lon_center, lat, lon)
                    for lat, lon in acc.geo_points
                ]
                avg_radius = sum(distances) / len(distances)
                p50_radius = _quantile(distances, 0.5)
                p90_radius = _quantile(distances, 0.9)

            category_top = [
                {"category": cat, "count": cnt}
                for cat, cnt in acc.category_counter.most_common(20)
            ]
            unique_categories = len(acc.category_counter)
            category_diversity = (
                unique_categories / interaction_total if interaction_total > 0 else 0.0
            )

            if acc.price_counter:
                dominant_price_level = max(
                    acc.price_counter.items(), key=lambda x: x[1]
                )[0]
            else:
                dominant_price_level = ""

            avg_business_stars = (
                acc.business_stars_sum / acc.business_quality_count
                if acc.business_quality_count > 0
                else 0.0
            )
            avg_business_review_log = (
                acc.business_review_log_sum / acc.business_quality_count
                if acc.business_quality_count > 0
                else 0.0
            )

            avg_business_checkins = (
                acc.business_checkin_total_sum / acc.business_checkin_count
                if acc.business_checkin_count > 0
                else 0.0
            )

            profile = {
                "user_id": uid,
                "profile_version": "rule_based_v1",
                "support": {
                    "support_level": _support_level(interaction_total),
                    "interaction_evidence_count": interaction_total,
                    "review_evidence_count": acc.review_count,
                    "tip_evidence_count": acc.tip_count,
                },
                "coverage": {
                    "unique_business_count": len(acc.business_ids),
                    "has_reviews": acc.review_count > 0,
                    "has_tips": acc.tip_count > 0,
                    "has_checkin_context": acc.business_checkin_count > 0,
                },
                "static": static,
                "rating_behavior": {
                    "avg_review_stars": round(avg_review_stars, 6),
                    "std_review_stars": round(std_review_stars, 6),
                    "positive_ratio": round(positive_ratio, 6),
                    "negative_ratio": round(negative_ratio, 6),
                    "avg_useful_per_review": round(
                        acc.useful_sum / rated, 6
                    )
                    if rated > 0
                    else 0.0,
                    "avg_funny_per_review": round(acc.funny_sum / rated, 6)
                    if rated > 0
                    else 0.0,
                    "avg_cool_per_review": round(acc.cool_sum / rated, 6)
                    if rated > 0
                    else 0.0,
                    "tip_compliment_sum": acc.tip_compliment_sum,
                },
                "temporal_pref": {
                    "hour_hist_24": acc.hour_hist,
                    "hour_pref_24": _normalize_hist(acc.hour_hist),
                    "weekday_hist_7": acc.weekday_hist,
                    "weekday_pref_7": _normalize_hist(acc.weekday_hist),
                    "weekend_ratio": round(weekend_ratio, 6),
                    "active_hours_top3": _top_indices(acc.hour_hist, 3),
                    "active_weekdays_top2": _top_indices(acc.weekday_hist, 2),
                },
                "category_pref": {
                    "top_categories": category_top,
                    "unique_category_count": unique_categories,
                    "category_diversity": round(category_diversity, 6),
                },
                "price_pref": {
                    "price_level_hist": dict(acc.price_counter),
                    "dominant_price_level": dominant_price_level,
                },
                "spatial_pref": {
                    "activity_center": {
                        "lat": round(lat_center, 6) if lat_center is not None else None,
                        "lon": round(lon_center, 6) if lon_center is not None else None,
                    },
                    "geo_point_count": len(acc.geo_points),
                    "avg_radius_km": round(avg_radius, 6) if avg_radius is not None else None,
                    "radius_km_p50": round(p50_radius, 6) if p50_radius is not None else None,
                    "radius_km_p90": round(p90_radius, 6) if p90_radius is not None else None,
                },
                "business_quality_pref": {
                    "avg_business_stars": round(avg_business_stars, 6),
                    "avg_business_review_count_log1p": round(avg_business_review_log, 6),
                },
                "checkin_context_pref": {
                    "avg_business_checkins": round(avg_business_checkins, 6),
                    "checkin_hour_hist_24": acc.checkin_hour_hist,
                    "checkin_hour_pref_24": _normalize_hist(acc.checkin_hour_hist),
                    "checkin_weekday_hist_7": acc.checkin_weekday_hist,
                    "checkin_weekday_pref_7": _normalize_hist(acc.checkin_weekday_hist),
                },
            }
            fw.write(json.dumps(profile, ensure_ascii=False) + "\n")

    print(f"Profile file written: {output_path}")


if __name__ == "__main__":
    main()
