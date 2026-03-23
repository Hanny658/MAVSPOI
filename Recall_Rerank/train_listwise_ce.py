from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
class TrainPaths:
    business_path: Path
    review_path: Path
    tip_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Recall+Rerank Non-LLM baseline with listwise cross-entropy."
    )
    parser.add_argument(
        "--train-source",
        choices=["train_side", "query_file"],
        default="train_side",
        help=(
            "Training sample source. "
            "'train_side' builds samples from train review/tip interactions (default). "
            "'query_file' uses prebuilt query/candidate files."
        ),
    )
    parser.add_argument(
        "--train-queries",
        default="data/eval/yelp-indianapolis-eval-queries.jsonl",
        help="Path to training query file with ground truth (used when --train-source=query_file).",
    )
    parser.add_argument(
        "--train-candidates",
        default="data/eval/yelp-indianapolis-eval-candidates.jsonl",
        help=(
            "Path to candidate file mapped by query_id "
            "(used when --train-source=query_file and --mode=constrained)."
        ),
    )
    parser.add_argument(
        "--train-business",
        default="",
        help="Optional train business path (default derived from settings).",
    )
    parser.add_argument(
        "--train-review",
        default="",
        help="Optional train review path (default derived from train business path).",
    )
    parser.add_argument(
        "--train-tip",
        default="",
        help="Optional train tip path (default derived from train business path).",
    )
    parser.add_argument(
        "--mode",
        choices=["constrained", "full"],
        default="constrained",
        help="Use candidate-constrained retrieval or full-corpus retrieval for training groups.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=500,
        help="Maximum training queries/groups to build. 0 means all.",
    )
    parser.add_argument(
        "--max-interactions-per-user",
        type=int,
        default=3,
        help=(
            "For train_side source, cap sampled interactions per user to reduce long-tail dominance. "
            "0 means no cap."
        ),
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=100,
        help="For train_side + constrained mode, candidate set size per synthetic query.",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=0.5,
        help="For train_side + constrained mode, ratio of hard negatives.",
    )
    parser.add_argument(
        "--model-out",
        default="Recall_Rerank/models/listwise_ce_model.json",
        help="Output model path.",
    )
    parser.add_argument(
        "--rerank-pool-size",
        type=int,
        default=0,
        help="Optional rerank pool size override.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


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


def _load_candidate_map(path: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for row in _iter_jsonl(path):
        qid = str(row.get("query_id", "")).strip()
        ids = row.get("candidate_business_ids", [])
        if not qid or not isinstance(ids, list):
            continue
        mapping[qid] = [str(x).strip() for x in ids if str(x).strip()]
    return mapping


def _to_context(row: dict[str, Any]):
    from src.schemas import GeoPoint, UserQueryContext

    loc = row.get("query_location") or {}
    lat = loc.get("lat")
    lon = loc.get("lon")
    geo = None
    if lat is not None and lon is not None:
        geo = GeoPoint(lat=float(lat), lon=float(lon))
    slice_info = row.get("evaluation_slice") or {}
    return UserQueryContext(
        query_text=str(row.get("query_text", "")).strip(),
        local_time=str(row.get("query_local_time", "")).strip(),
        location=geo,
        city=str(slice_info.get("city", "")).strip(),
        state=str(slice_info.get("state", "")).strip(),
        user_id=str(row.get("user_id", "anonymous")).strip() or "anonymous",
        long_term_notes="",
        recent_activity_notes="",
    )


def _parse_categories(raw: Any) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _derive_sibling(path: Path, src_key: str, dst_key: str) -> Path:
    name = path.name
    if src_key in name:
        return path.with_name(name.replace(src_key, dst_key))
    return path.with_name(name.replace("business", dst_key))


def _resolve_train_paths(
    settings: Any,
    business_override: str,
    review_override: str,
    tip_override: str,
) -> TrainPaths:
    business_path = Path(business_override.strip()) if business_override.strip() else Path(settings.yelp_business_json)
    review_path = Path(review_override.strip()) if review_override.strip() else _derive_sibling(
        business_path, "-business", "-review"
    )
    tip_path = Path(tip_override.strip()) if tip_override.strip() else _derive_sibling(
        business_path, "-business", "-tip"
    )
    if not business_path.exists():
        raise FileNotFoundError(f"Train business not found: {business_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Train review not found: {review_path}")
    if not tip_path.exists():
        raise FileNotFoundError(f"Train tip not found: {tip_path}")
    return TrainPaths(
        business_path=business_path,
        review_path=review_path,
        tip_path=tip_path,
    )


def _load_business_lite(path: Path) -> dict[str, BusinessLite]:
    output: dict[str, BusinessLite] = {}
    for row in _iter_jsonl(path):
        bid = str(row.get("business_id", "")).strip()
        if not bid:
            continue
        try:
            lat = float(row.get("latitude", 0.0))
            lon = float(row.get("longitude", 0.0))
        except (TypeError, ValueError):
            lat = 0.0
            lon = 0.0
        output[bid] = BusinessLite(
            business_id=bid,
            name=str(row.get("name", "")).strip(),
            city=str(row.get("city", "")).strip(),
            state=str(row.get("state", "")).strip(),
            latitude=lat,
            longitude=lon,
            categories=_parse_categories(row.get("categories")),
        )
    if not output:
        raise ValueError(f"No businesses loaded from: {path}")
    return output


def _category_hint(categories: list[str]) -> str:
    generic = {
        "restaurants",
        "food",
        "nightlife",
        "bars",
        "shopping",
        "local services",
        "coffee & tea",
    }
    for cat in categories:
        low = cat.strip().lower()
        if low and low not in generic:
            return cat.strip()
    return categories[0].strip() if categories else "place"


def _meal_hint(date_text: str) -> str:
    if len(date_text) >= 13 and date_text[11:13].isdigit():
        hour = int(date_text[11:13])
    else:
        hour = -1
    if 5 <= hour <= 10:
        return "breakfast"
    if 11 <= hour <= 14:
        return "lunch"
    if 17 <= hour <= 21:
        return "dinner"
    return "now"


def _synthetic_query_text(interaction_text: str, business: BusinessLite, date_text: str) -> str:
    cat = _category_hint(business.categories)
    meal = _meal_hint(date_text)
    text = (interaction_text or "").strip().lower()
    if any(tok in text for tok in ["quiet", "calm", "study", "work", "wifi"]):
        return f"Looking for a quiet {cat} spot with good vibe near me for {meal}"
    if any(tok in text for tok in ["family", "kids", "children"]):
        return f"Need a family friendly {cat} place near me for {meal}"
    if any(tok in text for tok in ["drink", "bar", "happy hour", "cocktail"]):
        return f"Find a good {cat} place for drinks near me tonight"
    if any(tok in text for tok in ["quick", "fast", "takeout", "delivery"]):
        return f"Need quick {cat} near me right now"
    return f"Find a good {cat} place near me for {meal}"


def _iter_train_side_rows(
    business_map: dict[str, BusinessLite],
    review_path: Path,
    tip_path: Path,
    max_interactions_per_user: int,
) -> Iterable[dict[str, Any]]:
    user_count: dict[str, int] = {}

    def _allow_user(uid: str) -> bool:
        if max_interactions_per_user <= 0:
            return True
        current = int(user_count.get(uid, 0))
        if current >= max_interactions_per_user:
            return False
        user_count[uid] = current + 1
        return True

    idx = 0
    for source, path in (("review", review_path), ("tip", tip_path)):
        for row in _iter_jsonl(path):
            user_id = str(row.get("user_id", "")).strip()
            business_id = str(row.get("business_id", "")).strip()
            date_text = str(row.get("date", "")).strip()
            if not user_id or not business_id or not date_text:
                continue
            business = business_map.get(business_id)
            if business is None:
                continue
            if not _allow_user(user_id):
                continue
            query_id = f"train-{source}-{idx}"
            idx += 1
            query_text = _synthetic_query_text(
                interaction_text=str(row.get("text", "")),
                business=business,
                date_text=date_text,
            )
            yield {
                "query_id": query_id,
                "user_id": user_id,
                "query_text": query_text,
                "query_local_time": date_text,
                "query_location": {
                    "lat": business.latitude,
                    "lon": business.longitude,
                },
                "ground_truth": {"business_id": business_id},
                "evaluation_slice": {
                    "city": business.city,
                    "state": business.state,
                },
                "candidate_business_ids": None,
            }


def _build_candidate_index(
    businesses: dict[str, BusinessLite],
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
    city_index: dict[str, list[str]] = {}
    category_index: dict[str, list[str]] = {}
    all_ids: list[str] = []
    for bid, b in businesses.items():
        all_ids.append(bid)
        city_key = b.city.lower()
        city_index.setdefault(city_key, []).append(bid)
        for cat in b.categories:
            key = cat.lower()
            category_index.setdefault(key, []).append(bid)
    return city_index, category_index, all_ids


def _make_candidate_set(
    gt_business_id: str,
    business: BusinessLite,
    city_index: dict[str, list[str]],
    category_index: dict[str, list[str]],
    all_business_ids: list[str],
    candidate_size: int,
    hard_negative_ratio: float,
    rng: random.Random,
) -> list[str]:
    target_size = max(2, int(candidate_size))
    hard_target = int((target_size - 1) * max(0.0, min(1.0, hard_negative_ratio)))
    picked = [gt_business_id]
    picked_set = {gt_business_id}

    city_pool = city_index.get(business.city.lower(), [])
    hard_pool: list[str] = []
    for cat in business.categories:
        for bid in category_index.get(cat.lower(), []):
            if bid == gt_business_id or bid in picked_set:
                continue
            if bid in city_pool:
                hard_pool.append(bid)
    rng.shuffle(hard_pool)
    for bid in hard_pool:
        if len(picked) >= 1 + hard_target:
            break
        if bid not in picked_set:
            picked.append(bid)
            picked_set.add(bid)

    medium_pool = [bid for bid in city_pool if bid not in picked_set]
    rng.shuffle(medium_pool)
    for bid in medium_pool:
        if len(picked) >= target_size:
            break
        picked.append(bid)
        picked_set.add(bid)

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


def main() -> None:
    args = parse_args()
    from Recall_Rerank.features import FEATURE_NAMES, build_feature_matrix
    from Recall_Rerank.model import ListwiseGroup, train_linear_listwise_ce
    from Recall_Rerank.pipeline import RecallRerankRealtimeRecommender
    from src.config import load_settings

    settings = load_settings()
    rng = random.Random(int(args.seed))
    recommender = RecallRerankRealtimeRecommender(
        settings=settings,
        model_path="",
        rerank_pool_size=(args.rerank_pool_size if args.rerank_pool_size > 0 else None),
    )

    train_rows: Iterable[dict[str, Any]]
    candidate_map: dict[str, list[str]] = {}
    resolved_train_paths: dict[str, str] = {}
    if args.train_source == "query_file":
        train_queries_path = Path(args.train_queries)
        train_candidates_path = Path(args.train_candidates)
        if not train_queries_path.exists():
            raise FileNotFoundError(f"Train queries not found: {train_queries_path}")
        if args.mode == "constrained" and not train_candidates_path.exists():
            raise FileNotFoundError(f"Train candidates not found: {train_candidates_path}")
        candidate_map = _load_candidate_map(train_candidates_path) if args.mode == "constrained" else {}
        train_rows = _iter_jsonl(train_queries_path)
        resolved_train_paths = {
            "train_queries": str(train_queries_path),
            "train_candidates": str(train_candidates_path) if args.mode == "constrained" else "",
        }
    else:
        paths = _resolve_train_paths(
            settings=settings,
            business_override=args.train_business,
            review_override=args.train_review,
            tip_override=args.train_tip,
        )
        business_map = _load_business_lite(paths.business_path)
        train_rows = _iter_train_side_rows(
            business_map=business_map,
            review_path=paths.review_path,
            tip_path=paths.tip_path,
            max_interactions_per_user=max(0, int(args.max_interactions_per_user)),
        )
        resolved_train_paths = {
            "train_business": str(paths.business_path),
            "train_review": str(paths.review_path),
            "train_tip": str(paths.tip_path),
        }
        if args.mode == "constrained":
            city_index, category_index, all_ids = _build_candidate_index(business_map)

            def _rows_with_candidates(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
                for row in rows:
                    gt_id = str((row.get("ground_truth") or {}).get("business_id", "")).strip()
                    business = business_map.get(gt_id)
                    if not gt_id or business is None:
                        continue
                    row["candidate_business_ids"] = _make_candidate_set(
                        gt_business_id=gt_id,
                        business=business,
                        city_index=city_index,
                        category_index=category_index,
                        all_business_ids=all_ids,
                        candidate_size=int(args.candidate_size),
                        hard_negative_ratio=float(args.hard_negative_ratio),
                        rng=rng,
                    )
                    yield row

            train_rows = _rows_with_candidates(train_rows)

    rerank_pool_k = max(
        settings.final_top_k,
        (args.rerank_pool_size if args.rerank_pool_size > 0 else settings.forecaster_top_k),
    )

    groups: list[ListwiseGroup] = []
    processed = 0
    skipped_no_gt = 0
    skipped_gt_not_in_pool = 0
    for row in train_rows:
        query_id = str(row.get("query_id", "")).strip()
        gt = row.get("ground_truth") or {}
        gt_id = str(gt.get("business_id", "")).strip()
        if not query_id or not gt_id:
            skipped_no_gt += 1
            continue

        context = _to_context(row)
        enriched_context, profile_features = recommender._build_enriched_context(context)
        if args.mode == "constrained":
            candidate_ids = row.get("candidate_business_ids")
            if not isinstance(candidate_ids, list):
                candidate_ids = candidate_map.get(query_id, [])
            initial_candidates = recommender.retriever.retrieve_from_pool(
                context=enriched_context,
                candidate_business_ids=[str(x).strip() for x in candidate_ids if str(x).strip()],
                top_k=settings.retrieval_top_k,
            )
        else:
            initial_candidates = recommender.retriever.retrieve(
                context=enriched_context,
                top_k=settings.retrieval_top_k,
            )

        rerank_pool = initial_candidates[:rerank_pool_k]
        gt_index = next(
            (
                idx
                for idx, candidate in enumerate(rerank_pool)
                if candidate.business.business_id == gt_id
            ),
            -1,
        )
        if gt_index < 0:
            skipped_gt_not_in_pool += 1
            continue

        features = build_feature_matrix(
            context=enriched_context,
            candidates=rerank_pool,
            profile_features=profile_features,
        )
        groups.append(
            ListwiseGroup(
                query_id=query_id,
                features=features,
                target_index=gt_index,
            )
        )
        processed += 1
        if args.max_queries > 0 and processed >= args.max_queries:
            break

    if not groups:
        raise RuntimeError("No valid training groups built. Cannot train model.")

    model, train_summary = train_linear_listwise_ce(
        groups=groups,
        feature_names=FEATURE_NAMES,
        lr=float(args.lr),
        epochs=int(args.epochs),
        l2=float(args.l2),
        seed=int(args.seed),
    )
    model.metadata["mode"] = args.mode
    model.metadata["train_source"] = args.train_source
    model.metadata["rerank_pool_size"] = int(rerank_pool_k)
    for key, value in resolved_train_paths.items():
        model.metadata[key] = value
    model.save(args.model_out)

    summary = {
        "mode": args.mode,
        "train_source": args.train_source,
        "model_out": str(args.model_out),
        "rerank_pool_size": int(rerank_pool_k),
        "processed_groups": int(len(groups)),
        "skipped_no_gt": int(skipped_no_gt),
        "skipped_gt_not_in_pool": int(skipped_gt_not_in_pool),
        "feature_count": len(FEATURE_NAMES),
        "train_paths": resolved_train_paths,
        "train": train_summary,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
