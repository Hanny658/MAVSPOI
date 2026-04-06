from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs) -> bool:
        return False

from openai import OpenAI


@dataclass
class BusinessLite:
    business_id: str
    name: str
    city: str
    state: str
    latitude: float
    longitude: float
    categories: list[str]


@dataclass
class VisitEvent:
    user_id: str
    business_id: str
    date_text: str
    timestamp: datetime
    source: str
    line_no: int


def parse_args() -> argparse.Namespace:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(
        description=(
            "Build a train/test protocol by holding out each user's latest visit as ground truth, "
            "then generating query/candidate files and rebuilding train-side user profiles."
        )
    )
    parser.add_argument(
        "--data-prefix",
        default="yelp-indianapolis",
        help="Input subset prefix. Expected files: data/<prefix>-*.jsonl or .json",
    )
    parser.add_argument(
        "--train-prefix",
        default="",
        help="Output train prefix. Default: <data-prefix>-train",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory path (default: data).",
    )
    parser.add_argument(
        "--train-dir",
        default="train",
        help="Train output subdirectory under data-dir (default: train).",
    )
    parser.add_argument(
        "--eval-dir",
        default="eval",
        help="Evaluation output subdirectory under data-dir (default: eval).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help=(
            "Randomly sample this many users for evaluation holdout. "
            "0 means use all users with visits. Default: 500."
        ),
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help=(
            "Deprecated alias for --sample-size. "
            "Only used when --sample-size is 0."
        ),
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=100,
        help="Candidate set size per query (default: 100).",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=0.5,
        help="Ratio of hard negatives from same city/category (default: 0.5).",
    )
    parser.add_argument(
        "--balance-support",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to balance sampled eval users across zero_shot/few_shot/warm "
            "based on support level after holdout (default: true)."
        ),
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.getenv("OPENAI_BASE_URL", "").strip(),
        help="Optional OpenAI-compatible base URL. Empty means official OpenAI cloud endpoint.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=1.1,
        help="Sampling temperature for diverse cloud query generation.",
    )
    parser.add_argument(
        "--llm-top-p",
        type=float,
        default=0.95,
        help="Top-p sampling for diverse cloud query generation.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=3,
        help="Maximum retries per query when cloud generation fails.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic protocol generation.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200000,
        help="Progress log frequency while scanning large files.",
    )
    return parser.parse_args()


def resolve_path(data_dir: Path, prefix: str, suffix: str) -> Path:
    for ext in (".jsonl", ".json"):
        path = data_dir / f"{prefix}-{suffix}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Cannot find {prefix}-{suffix}.jsonl/.json under {data_dir}"
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_datetime(date_text: str) -> datetime | None:
    date_text = (date_text or "").strip()
    if not date_text:
        return None
    # Yelp uses "YYYY-MM-DD HH:MM:SS".
    try:
        return datetime.strptime(date_text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_categories(raw: Any) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def support_level_after_holdout(remaining_visits: int) -> str:
    if remaining_visits <= 0:
        return "zero_shot"
    if remaining_visits <= 10:
        return "few_shot"
    return "warm"


def sample_users_by_support_balance(
    latest: dict[str, VisitEvent],
    visit_counts: dict[str, int],
    sample_limit: int,
    rng: random.Random,
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    order = ["zero_shot", "few_shot", "warm"]
    buckets: dict[str, list[str]] = {k: [] for k in order}
    for uid in latest.keys():
        remaining = max(0, int(visit_counts.get(uid, 0)) - 1)
        level = support_level_after_holdout(remaining)
        buckets.setdefault(level, []).append(uid)
    for level in buckets:
        rng.shuffle(buckets[level])

    eligible_counts = {level: len(buckets.get(level, [])) for level in order}
    total_eligible = sum(eligible_counts.values())
    if sample_limit <= 0 or sample_limit > total_eligible:
        sample_limit = total_eligible

    selected: list[str] = []
    selected_counts: Counter[str] = Counter()
    while len(selected) < sample_limit:
        non_empty = [level for level in order if buckets.get(level)]
        if not non_empty:
            break
        min_selected = min(selected_counts[level] for level in non_empty)
        candidates = [level for level in non_empty if selected_counts[level] == min_selected]
        candidates.sort(key=lambda level: len(buckets[level]), reverse=True)
        chosen_level = candidates[0]
        uid = buckets[chosen_level].pop()
        selected.append(uid)
        selected_counts[chosen_level] += 1

    return selected, dict(selected_counts), eligible_counts


def perturb_location(
    lat: float,
    lon: float,
    distance_m: float,
    bearing_rad: float,
) -> tuple[float, float]:
    # Great-circle forward approximation for small distances.
    radius_m = 6_371_000.0
    d = distance_m / radius_m
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d)
        + math.cos(lat1) * math.sin(d) * math.cos(bearing_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing_rad) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def category_hint(categories: list[str]) -> str:
    generic = {
        "restaurants",
        "food",
        "nightlife",
        "bars",
        "shopping",
        "local services",
    }
    for c in categories:
        cl = c.strip().lower()
        if cl and cl not in generic:
            return c.strip()
    return categories[0].strip() if categories else "place"


def meal_period_from_hour(hour: int) -> str:
    if 5 <= hour <= 10:
        return "breakfast"
    if 11 <= hour <= 14:
        return "lunch"
    if 17 <= hour <= 21:
        return "dinner"
    return "late-night"


class CloudLLMQueryGenerator:
    PROMPT_PROFILES = [
        {
            "name": "concise_search",
            "style_instruction": (
                "Write like a concise mobile search query. Keep it direct and practical."
            ),
            "ambiguity_instruction": "Keep one clear intent but leave optional preferences implicit.",
        },
        {
            "name": "conversational_request",
            "style_instruction": (
                "Write like a natural spoken request to an assistant, with colloquial phrasing."
            ),
            "ambiguity_instruction": "Be moderately ambiguous: mention need, not exact constraints.",
        },
        {
            "name": "decision_anxiety",
            "style_instruction": (
                "Write like a user undecided between options, expressing uncertainty."
            ),
            "ambiguity_instruction": "High ambiguity: keep category broad and constraints loose.",
        },
        {
            "name": "task_oriented",
            "style_instruction": (
                "Write like task-driven intent (what user wants to do), not only category words."
            ),
            "ambiguity_instruction": "Medium ambiguity: include context but not strict filters.",
        },
        {
            "name": "time_pressure",
            "style_instruction": (
                "Write like the user is in a hurry and wants a quick recommendation."
            ),
            "ambiguity_instruction": "Low ambiguity on urgency, high ambiguity on exact venue type.",
        },
        {
            "name": "vibe_preference",
            "style_instruction": (
                "Write like the user cares about vibe/experience (quiet, cozy, lively, etc.)."
            ),
            "ambiguity_instruction": "Keep constraints soft and experiential.",
        },
    ]
    QUERY_FAMILY_RATIOS = {
        "explicit_category": 0.15,
        "soft_constraint": 0.35,
        "generic_need": 0.50,
    }
    SOFT_HINT_MAP = {
        "spicy": {
            "mexican",
            "indian",
            "thai",
            "korean",
            "szechuan",
            "cajun",
            "hot pot",
        },
        "hearty_portion": {
            "bbq",
            "barbeque",
            "steakhouses",
            "burgers",
            "pizza",
            "sandwiches",
            "southern",
        },
        "budget_friendly": {
            "fast food",
            "food trucks",
            "diners",
            "pizza",
            "sandwiches",
        },
        "quick_service": {
            "fast food",
            "food trucks",
            "takeout",
            "delis",
        },
        "light_or_healthy": {
            "vegan",
            "vegetarian",
            "salad",
            "juice bars & smoothies",
            "mediterranean",
            "poke",
        },
        "cozy_or_chill": {
            "cafes",
            "coffee & tea",
            "tea rooms",
            "bakeries",
        },
    }

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float,
        top_p: float,
        max_retries: int,
        rng: random.Random,
        expected_queries: int | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError(
                "Cloud API key is required. Set OPENAI_API_KEY in environment."
            )
        if base_url.strip():
            self.client = OpenAI(base_url=base_url.strip(), api_key=api_key.strip())
        else:
            self.client = OpenAI(api_key=api_key.strip())
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max(1, max_retries)
        self.calls = 0
        self.rng = rng
        self.expected_queries = (
            max(1, int(expected_queries))
            if expected_queries is not None and int(expected_queries) > 0
            else None
        )
        self.family_counts: Counter[str] = Counter()
        self.family_targets = self._build_family_targets(self.expected_queries)

    def _build_family_targets(self, expected_queries: int | None) -> dict[str, int]:
        if not expected_queries:
            return {}
        explicit = int(round(expected_queries * self.QUERY_FAMILY_RATIOS["explicit_category"]))
        soft = int(round(expected_queries * self.QUERY_FAMILY_RATIOS["soft_constraint"]))
        generic = max(0, expected_queries - explicit - soft)
        return {
            "explicit_category": explicit,
            "soft_constraint": soft,
            "generic_need": generic,
        }

    def _select_query_family(self) -> str:
        if self.family_targets:
            remaining = {
                family: target - int(self.family_counts.get(family, 0))
                for family, target in self.family_targets.items()
            }
            available = {k: v for k, v in remaining.items() if v > 0}
            if available:
                families = list(available.keys())
                weights = [float(available[f]) for f in families]
                return str(self.rng.choices(families, weights=weights, k=1)[0])
        families = list(self.QUERY_FAMILY_RATIOS.keys())
        weights = [float(self.QUERY_FAMILY_RATIOS[f]) for f in families]
        return str(self.rng.choices(families, weights=weights, k=1)[0])

    def _build_soft_hints(self, categories: list[str]) -> list[str]:
        cat_lower = [str(c).strip().lower() for c in categories if str(c).strip()]
        hints: list[str] = []
        for hint, trigger_categories in self.SOFT_HINT_MAP.items():
            for cat in cat_lower:
                if cat in trigger_categories:
                    hints.append(hint)
                    break
        if not hints:
            hints = ["comfort_food", "satisfying", "good_value"]
        self.rng.shuffle(hints)
        return hints[:3]

    def _profile_for_family(self, family: str) -> dict[str, str]:
        pool = list(self.PROMPT_PROFILES)
        if family == "explicit_category":
            filtered = [
                p for p in pool if p["name"] in {"concise_search", "conversational_request"}
            ]
            pool = filtered or pool
        elif family == "soft_constraint":
            filtered = [
                p for p in pool if p["name"] in {"vibe_preference", "task_oriented", "decision_anxiety"}
            ]
            pool = filtered or pool
        else:
            filtered = [
                p for p in pool if p["name"] in {"decision_anxiety", "conversational_request", "time_pressure"}
            ]
            pool = filtered or pool
        return self.rng.choice(pool)

    def generate(
        self,
        event: VisitEvent,
        business: BusinessLite,
        query_time: datetime,
        remaining_visits: int,
    ) -> tuple[str, str, str]:
        cat_hint = category_hint(business.categories)
        support = support_level_after_holdout(remaining_visits)
        meal_hint = meal_period_from_hour(query_time.hour)
        last_error = "unknown_error"
        family = self._select_query_family()

        for _ in range(self.max_retries):
            profile = self._profile_for_family(family)
            if family == "explicit_category":
                family_instruction = (
                    "Query family: explicit_category (15%).\n"
                    "You may use ONLY the injected category hint from GT.\n"
                    "Do not add any other GT-derived clues (city/state, popularity, ratings, specific dishes, or store traits).\n"
                    "The query should clearly mention that category and remain short."
                )
                user_payload = {
                    "query_family": family,
                    "time_local": query_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "hour": query_time.hour,
                    "meal_period_hint": meal_hint,
                    "allowed_gt_field": {"category_hint": cat_hint},
                    "query_goal": "explicit category seeking request near me",
                }
            elif family == "soft_constraint":
                soft_hints = self._build_soft_hints(business.categories)
                family_instruction = (
                    "Query family: soft_constraint (35%).\n"
                    "You can use GT-derived latent traits, but NEVER mention cuisine/category terms.\n"
                    "Write soft-constraint style needs such as affordable, spicy, filling, cozy, quick, etc.\n"
                    "Do not reveal explicit category words or business identity clues."
                )
                user_payload = {
                    "query_family": family,
                    "time_local": query_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "hour": query_time.hour,
                    "meal_period_hint": meal_hint,
                    "city": business.city,
                    "state": business.state,
                    "soft_hints_from_gt": soft_hints,
                    "support_level": support,
                    "remaining_visits_after_holdout": remaining_visits,
                    "query_goal": "soft-constraint food request without explicit category",
                }
            else:
                family_instruction = (
                    "Query family: generic_need (50%).\n"
                    "No GT information is available in this mode.\n"
                    "Generate very generic and broad food-seeking queries (e.g., user is unsure what to eat now).\n"
                    "Do not mention explicit cuisine/category terms."
                )
                user_payload = {
                    "query_family": family,
                    "time_local": query_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "hour": query_time.hour,
                    "meal_period_hint": meal_hint,
                    "support_level": support,
                    "remaining_visits_after_holdout": remaining_visits,
                    "query_goal": "highly generic request for food recommendation near me",
                }

            system_prompt = (
                "You generate realistic user queries for a query-based POI recommender.\n"
                "Hard constraints:\n"
                "- Do NOT mention business name/address/coordinates/business_id.\n"
                "- Use natural human wording, avoid rigid templates.\n"
                "- Keep query length between 6 and 28 words.\n"
                "- Mention near-me/location context naturally.\n"
                "- Output JSON only: {\"query\": \"...\", \"style\": \"...\", \"family\": \"...\"}.\n"
                f"{family_instruction}\n"
                f"Style profile: {profile['name']}\n"
                f"Style instruction: {profile['style_instruction']}\n"
                f"Ambiguity instruction: {profile['ambiguity_instruction']}\n"
            )
            try:
                self.calls += 1
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": json.dumps(user_payload, ensure_ascii=False),
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                raw = response.choices[0].message.content or "{}"
                if isinstance(raw, list):
                    raw = "".join(
                        item.get("text", "") for item in raw if isinstance(item, dict)
                    )
                row = json.loads(raw)
                query = str(row.get("query", "")).strip()
                if not query:
                    last_error = "empty_query"
                    continue

                self.family_counts[family] += 1
                style_name = str(row.get("style", "")).strip() or str(profile["name"])
                style_key = f"{family}:{style_name}"
                return query, style_key, family
            except Exception as exc:
                last_error = f"api_error:{type(exc).__name__}"
                continue

        raise RuntimeError(
            "Failed to generate query from cloud LLM after retries. "
            f"last_error={last_error}"
        )


def load_businesses(path: Path) -> dict[str, BusinessLite]:
    businesses: dict[str, BusinessLite] = {}
    for row in iter_jsonl(path):
        bid = str(row.get("business_id", "")).strip()
        if not bid:
            continue
        businesses[bid] = BusinessLite(
            business_id=bid,
            name=str(row.get("name", "")).strip(),
            city=str(row.get("city", "")).strip(),
            state=str(row.get("state", "")).strip(),
            latitude=float(row.get("latitude", 0.0)),
            longitude=float(row.get("longitude", 0.0)),
            categories=parse_categories(row.get("categories")),
        )
    if not businesses:
        raise ValueError(f"No businesses loaded from: {path}")
    return businesses


def collect_last_visits(
    review_path: Path,
    tip_path: Path,
    valid_business_ids: set[str],
    progress_every: int,
) -> tuple[dict[str, VisitEvent], dict[str, int]]:
    user_counts: dict[str, int] = Counter()
    latest: dict[str, VisitEvent] = {}

    for i, row in enumerate(iter_jsonl(review_path), 1):
        uid = str(row.get("user_id", "")).strip()
        bid = str(row.get("business_id", "")).strip()
        dt = parse_datetime(str(row.get("date", "")))
        if not uid or not bid or bid not in valid_business_ids or dt is None:
            continue
        user_counts[uid] += 1
        event = VisitEvent(
            user_id=uid,
            business_id=bid,
            date_text=str(row.get("date", "")).strip(),
            timestamp=dt,
            source="review",
            line_no=i,
        )
        prev = latest.get(uid)
        if prev is None or event.timestamp >= prev.timestamp:
            latest[uid] = event
        if i % progress_every == 0:
            print(f"  review scanned: {i:,}")

    for i, row in enumerate(iter_jsonl(tip_path), 1):
        uid = str(row.get("user_id", "")).strip()
        bid = str(row.get("business_id", "")).strip()
        dt = parse_datetime(str(row.get("date", "")))
        if not uid or not bid or bid not in valid_business_ids or dt is None:
            continue
        user_counts[uid] += 1
        event = VisitEvent(
            user_id=uid,
            business_id=bid,
            date_text=str(row.get("date", "")).strip(),
            timestamp=dt,
            source="tip",
            line_no=i,
        )
        prev = latest.get(uid)
        if prev is None or event.timestamp >= prev.timestamp:
            latest[uid] = event
        if i % progress_every == 0:
            print(f"  tip scanned: {i:,}")

    return latest, user_counts


def build_candidate_index(
    businesses: dict[str, BusinessLite],
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
    city_index: dict[str, list[str]] = defaultdict(list)
    category_index: dict[str, list[str]] = defaultdict(list)
    all_ids: list[str] = []
    for bid, b in businesses.items():
        all_ids.append(bid)
        city_key = b.city.lower()
        city_index[city_key].append(bid)
        for cat in b.categories:
            category_index[cat.lower()].append(bid)
    return city_index, category_index, all_ids


def make_candidate_set(
    gt_business_id: str,
    business: BusinessLite,
    city_index: dict[str, list[str]],
    category_index: dict[str, list[str]],
    all_business_ids: list[str],
    candidate_size: int,
    hard_negative_ratio: float,
    rng: random.Random,
) -> list[str]:
    target_size = max(2, candidate_size)
    hard_target = int((target_size - 1) * max(0.0, min(1.0, hard_negative_ratio)))
    picked = [gt_business_id]
    picked_set = {gt_business_id}

    # Hard negatives: same city and same category as much as possible.
    hard_pool: list[str] = []
    city_pool = city_index.get(business.city.lower(), [])
    category_pools = [category_index.get(c.lower(), []) for c in business.categories]
    for cat_pool in category_pools:
        for bid in cat_pool:
            if bid == gt_business_id:
                continue
            if bid in picked_set:
                continue
            # Keep mostly city-consistent hard negatives first.
            if bid in city_pool:
                hard_pool.append(bid)
    rng.shuffle(hard_pool)
    for bid in hard_pool:
        if len(picked) >= 1 + hard_target:
            break
        if bid not in picked_set:
            picked.append(bid)
            picked_set.add(bid)

    # Medium negatives: same city.
    medium_pool = [bid for bid in city_pool if bid not in picked_set]
    rng.shuffle(medium_pool)
    for bid in medium_pool:
        if len(picked) >= target_size:
            break
        picked.append(bid)
        picked_set.add(bid)

    # Easy negatives: global random fill.
    if len(picked) < target_size:
        global_pool = [bid for bid in all_business_ids if bid not in picked_set]
        rng.shuffle(global_pool)
        for bid in global_pool:
            if len(picked) >= target_size:
                break
            picked.append(bid)
            picked_set.add(bid)

    rng.shuffle(picked)
    return picked[:target_size]


def write_filtered_train_files(
    data_dir: Path,
    train_output_dir: Path,
    input_prefix: str,
    train_prefix: str,
    users_with_visits: set[str],
    drop_review_lines: set[int],
    drop_tip_lines: set[int],
) -> dict[str, int]:
    in_business = resolve_path(data_dir, input_prefix, "business")
    in_review = resolve_path(data_dir, input_prefix, "review")
    in_tip = resolve_path(data_dir, input_prefix, "tip")
    in_user = resolve_path(data_dir, input_prefix, "user")
    in_checkin = resolve_path(data_dir, input_prefix, "checkin")

    train_output_dir.mkdir(parents=True, exist_ok=True)
    out_business = train_output_dir / f"{train_prefix}-business.jsonl"
    out_review = train_output_dir / f"{train_prefix}-review.jsonl"
    out_tip = train_output_dir / f"{train_prefix}-tip.jsonl"
    out_user = train_output_dir / f"{train_prefix}-user.jsonl"
    out_checkin = train_output_dir / f"{train_prefix}-checkin.jsonl"

    stats = {
        "review_removed_as_gt": 0,
        "tip_removed_as_gt": 0,
        "review_out": 0,
        "tip_out": 0,
        "user_out": 0,
    }

    # Business and checkin are copied directly because ground-truth holdout is interaction-level.
    out_business.write_bytes(in_business.read_bytes())
    out_checkin.write_bytes(in_checkin.read_bytes())

    with in_review.open("r", encoding="utf-8") as fr, out_review.open(
        "w", encoding="utf-8"
    ) as fw:
        for i, line in enumerate(fr, 1):
            if i in drop_review_lines:
                stats["review_removed_as_gt"] += 1
                continue
            fw.write(line)
            stats["review_out"] += 1

    with in_tip.open("r", encoding="utf-8") as fr, out_tip.open("w", encoding="utf-8") as fw:
        for i, line in enumerate(fr, 1):
            if i in drop_tip_lines:
                stats["tip_removed_as_gt"] += 1
                continue
            fw.write(line)
            stats["tip_out"] += 1

    with in_user.open("r", encoding="utf-8") as fr, out_user.open("w", encoding="utf-8") as fw:
        for line in fr:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                row = json.loads(line_stripped)
            except json.JSONDecodeError:
                continue
            uid = str(row.get("user_id", "")).strip()
            if uid and uid in users_with_visits:
                fw.write(line if line.endswith("\n") else line + "\n")
                stats["user_out"] += 1

    return stats


def run_profile_builder(train_output_dir: Path, train_prefix: str) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "utils" / "build_user_profile.py"),
        "--data-prefix",
        train_prefix,
        "--data-dir",
        str(train_output_dir),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    train_output_dir = data_dir / args.train_dir
    eval_output_dir = data_dir / args.eval_dir
    train_output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    input_prefix = args.data_prefix.strip()
    train_prefix = args.train_prefix.strip() or f"{input_prefix}-train"

    business_path = resolve_path(data_dir, input_prefix, "business")
    review_path = resolve_path(data_dir, input_prefix, "review")
    tip_path = resolve_path(data_dir, input_prefix, "tip")
    user_path = resolve_path(data_dir, input_prefix, "user")
    _ = user_path  # Explicitly resolved to fail-fast if missing.

    query_output = eval_output_dir / f"{input_prefix}-eval-queries.jsonl"
    candidate_output = eval_output_dir / f"{input_prefix}-eval-candidates.jsonl"
    meta_output = eval_output_dir / f"{input_prefix}-eval-meta.json"
    query_tmp = eval_output_dir / f"{input_prefix}-eval-queries.jsonl.tmp"
    candidate_tmp = eval_output_dir / f"{input_prefix}-eval-candidates.jsonl.tmp"

    print(f"[1/7] Loading business map: {business_path}")
    businesses = load_businesses(business_path)
    print(f"  businesses loaded: {len(businesses):,}")

    print("[2/7] Scanning latest user visits from review/tip")
    latest, visit_counts = collect_last_visits(
        review_path=review_path,
        tip_path=tip_path,
        valid_business_ids=set(businesses.keys()),
        progress_every=args.progress_every,
    )
    users_with_visits = {u for u, c in visit_counts.items() if c > 0}
    if not latest:
        raise ValueError("No valid user visits found. Cannot build evaluation protocol.")
    print(
        f"  users_with_visits={len(users_with_visits):,}, "
        f"holdout_events={len(latest):,}"
    )

    city_index, category_index, all_business_ids = build_candidate_index(businesses)

    llm_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
    llm_api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("OPENAI_KEY", "").strip()
    )
    if not llm_api_key:
        raise ValueError(
            "Missing API key. Set OPENAI_API_KEY in environment or .env."
        )

    print("[3/7] Sampling evaluation users and building query/candidate files")
    all_users = list(latest.keys())

    # Prefer sample-size. max-users is kept for backward compatibility.
    sample_limit = args.sample_size if args.sample_size > 0 else 0
    if sample_limit <= 0 and args.max_users > 0:
        sample_limit = args.max_users
    if args.balance_support:
        selected_users, selected_support_counts, eligible_support_counts = (
            sample_users_by_support_balance(
                latest=latest,
                visit_counts=visit_counts,
                sample_limit=sample_limit,
                rng=rng,
            )
        )
    else:
        rng.shuffle(all_users)
        selected_users = all_users[:sample_limit] if sample_limit > 0 else all_users
        selected_support_counts = Counter()
        for uid in selected_users:
            remaining = max(0, int(visit_counts.get(uid, 0)) - 1)
            selected_support_counts[support_level_after_holdout(remaining)] += 1
        eligible_support_counts = Counter()
        for uid in all_users:
            remaining = max(0, int(visit_counts.get(uid, 0)) - 1)
            eligible_support_counts[support_level_after_holdout(remaining)] += 1

    selected_latest = {uid: latest[uid] for uid in selected_users}
    drop_review_lines = {
        event.line_no
        for event in selected_latest.values()
        if event.source == "review"
    }
    drop_tip_lines = {
        event.line_no
        for event in selected_latest.values()
        if event.source == "tip"
    }

    llm_generator = CloudLLMQueryGenerator(
        base_url=args.llm_base_url,
        model=llm_model,
        api_key=llm_api_key,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        max_retries=args.llm_max_retries,
        rng=rng,
        expected_queries=len(selected_users),
    )

    query_count = 0
    llm_success = 0
    style_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    with query_tmp.open("w", encoding="utf-8") as fq, candidate_tmp.open(
        "w", encoding="utf-8"
    ) as fc:
        for uid in selected_users:
            event = selected_latest[uid]
            b = businesses.get(event.business_id)
            if b is None:
                continue

            back_minutes = rng.randint(5, 30)
            query_time = event.timestamp - timedelta(minutes=back_minutes)
            move_meters = rng.uniform(100.0, 500.0)
            bearing = rng.uniform(0.0, 2.0 * math.pi)
            q_lat, q_lon = perturb_location(
                lat=b.latitude,
                lon=b.longitude,
                distance_m=move_meters,
                bearing_rad=bearing,
            )

            remaining = max(0, visit_counts[uid] - 1)
            query_text, style_name, family_name = llm_generator.generate(
                event=event,
                business=b,
                query_time=query_time,
                remaining_visits=remaining,
            )
            llm_success += 1
            style_counter[style_name] += 1
            family_counter[family_name] += 1

            query_id = f"{input_prefix}-{uid}-{event.source}-{event.line_no}"
            query_row = {
                "query_id": query_id,
                "user_id": uid,
                "query_text": query_text,
                "query_style": style_name,
                "query_family": family_name,
                "query_local_time": query_time.strftime("%Y-%m-%d %H:%M:%S"),
                "query_location": {
                    "lat": round(q_lat, 6),
                    "lon": round(q_lon, 6),
                },
                "ground_truth": {
                    "business_id": event.business_id,
                    "visit_time": event.date_text,
                    "source": event.source,
                },
                "perturbation": {
                    "time_back_minutes": back_minutes,
                    "distance_meters": round(move_meters, 3),
                },
                "evaluation_slice": {
                    "remaining_visits_after_holdout": remaining,
                    "support_level_after_holdout": support_level_after_holdout(remaining),
                    "city": b.city,
                    "state": b.state,
                },
            }
            fq.write(json.dumps(query_row, ensure_ascii=False) + "\n")

            candidate_ids = make_candidate_set(
                gt_business_id=event.business_id,
                business=b,
                city_index=city_index,
                category_index=category_index,
                all_business_ids=all_business_ids,
                candidate_size=args.candidate_size,
                hard_negative_ratio=args.hard_negative_ratio,
                rng=rng,
            )
            candidate_row = {
                "query_id": query_id,
                "user_id": uid,
                "ground_truth_business_id": event.business_id,
                "candidate_business_ids": candidate_ids,
            }
            fc.write(json.dumps(candidate_row, ensure_ascii=False) + "\n")

            query_count += 1
            if query_count % 50000 == 0:
                print(f"  queries built: {query_count:,}")

    # Atomic replace to avoid leaving corrupted output files after interrupted runs.
    query_tmp.replace(query_output)
    candidate_tmp.replace(candidate_output)

    print(f"  eval queries built: {query_count:,}")

    print("[4/7] Writing train split files with held-out interactions removed")
    split_stats = write_filtered_train_files(
        data_dir=data_dir,
        train_output_dir=train_output_dir,
        input_prefix=input_prefix,
        train_prefix=train_prefix,
        users_with_visits=users_with_visits,
        drop_review_lines=drop_review_lines,
        drop_tip_lines=drop_tip_lines,
    )

    print("[5/7] Rebuilding train-side user profiles <With sub-tasks>")
    run_profile_builder(train_output_dir, train_prefix)

    print("[6/7] Writing protocol metadata")
    meta = {
        "input_prefix": input_prefix,
        "train_prefix": train_prefix,
        "seed": args.seed,
        "query_generator": "cloud_llm_only",
        "llm_base_url": args.llm_base_url,
        "llm_model": llm_model,
        "llm_temperature": args.llm_temperature,
        "llm_top_p": args.llm_top_p,
        "llm_max_retries": args.llm_max_retries,
        "llm_api_call_count": llm_generator.calls,
        "llm_success_count": llm_success,
        "llm_failure_count": 0,
        "query_style_distribution": dict(style_counter),
        "query_family_distribution": dict(family_counter),
        "query_family_target_distribution": dict(llm_generator.family_targets),
        "candidate_size": args.candidate_size,
        "hard_negative_ratio": args.hard_negative_ratio,
        "users_with_visits": len(users_with_visits),
        "held_out_events": len(selected_latest),
        "sampling": {
            "all_eligible_users": len(all_users),
            "sample_size": len(selected_users),
            "seed": args.seed,
            "balance_support": bool(args.balance_support),
            "support_eligible_counts": dict(eligible_support_counts),
            "support_selected_counts": dict(selected_support_counts),
        },
        "eval_queries_count": query_count,
        "files": {
            "eval_queries": str(query_output),
            "eval_candidates": str(candidate_output),
            "train_business": str(train_output_dir / f"{train_prefix}-business.jsonl"),
            "train_review": str(train_output_dir / f"{train_prefix}-review.jsonl"),
            "train_tip": str(train_output_dir / f"{train_prefix}-tip.jsonl"),
            "train_user": str(train_output_dir / f"{train_prefix}-user.jsonl"),
            "train_checkin": str(train_output_dir / f"{train_prefix}-checkin.jsonl"),
            "train_profile": str(train_output_dir / f"{train_prefix}-profile.jsonl"),
        },
        "split_stats": split_stats,
    }
    with meta_output.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[7/7] Done")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
