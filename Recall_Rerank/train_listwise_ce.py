from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Recall+Rerank Non-LLM baseline with listwise cross-entropy."
    )
    parser.add_argument(
        "--train-queries",
        default="data/eval/yelp-indianapolis-eval-queries.jsonl",
        help="Path to training query file with ground truth.",
    )
    parser.add_argument(
        "--train-candidates",
        default="data/eval/yelp-indianapolis-eval-candidates.jsonl",
        help="Path to candidate file mapped by query_id.",
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
        default=0,
        help="Maximum training queries to build. 0 means all.",
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


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_candidate_map(path: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for row in _iter_jsonl(path):
        qid = str(row.get("query_id", "")).strip()
        ids = row.get("candidate_business_ids", [])
        if not qid or not isinstance(ids, list):
            continue
        mapping[qid] = [str(x).strip() for x in ids if str(x).strip()]
    return mapping


def _to_context(row: dict[str, Any]) -> UserQueryContext:
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


def main() -> None:
    args = parse_args()
    from Recall_Rerank.features import FEATURE_NAMES, build_feature_matrix
    from Recall_Rerank.model import ListwiseGroup, train_linear_listwise_ce
    from Recall_Rerank.pipeline import RecallRerankRealtimeRecommender
    from src.config import load_settings

    train_queries_path = Path(args.train_queries)
    train_candidates_path = Path(args.train_candidates)
    if not train_queries_path.exists():
        raise FileNotFoundError(f"Train queries not found: {train_queries_path}")
    if args.mode == "constrained" and not train_candidates_path.exists():
        raise FileNotFoundError(f"Train candidates not found: {train_candidates_path}")

    settings = load_settings()
    recommender = RecallRerankRealtimeRecommender(
        settings=settings,
        model_path="",
        rerank_pool_size=(args.rerank_pool_size if args.rerank_pool_size > 0 else None),
    )
    candidate_map = _load_candidate_map(train_candidates_path) if args.mode == "constrained" else {}
    rerank_pool_k = max(
        settings.final_top_k,
        (args.rerank_pool_size if args.rerank_pool_size > 0 else settings.forecaster_top_k),
    )

    groups: list[ListwiseGroup] = []
    processed = 0
    skipped_no_gt = 0
    skipped_gt_not_in_pool = 0
    for row in _iter_jsonl(train_queries_path):
        query_id = str(row.get("query_id", "")).strip()
        gt = row.get("ground_truth") or {}
        gt_id = str(gt.get("business_id", "")).strip()
        if not query_id or not gt_id:
            skipped_no_gt += 1
            continue

        context = _to_context(row)
        enriched_context, profile_features = recommender._build_enriched_context(context)
        if args.mode == "constrained":
            initial_candidates = recommender.retriever.retrieve_from_pool(
                context=enriched_context,
                candidate_business_ids=candidate_map.get(query_id, []),
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
    model.metadata["train_queries_path"] = str(train_queries_path)
    model.metadata["train_candidates_path"] = str(train_candidates_path) if args.mode == "constrained" else ""
    model.metadata["rerank_pool_size"] = int(rerank_pool_k)
    model.save(args.model_out)

    summary = {
        "mode": args.mode,
        "train_queries": str(train_queries_path),
        "train_candidates": str(train_candidates_path) if args.mode == "constrained" else "",
        "model_out": str(args.model_out),
        "rerank_pool_size": int(rerank_pool_k),
        "processed_groups": int(len(groups)),
        "skipped_no_gt": int(skipped_no_gt),
        "skipped_gt_not_in_pool": int(skipped_gt_not_in_pool),
        "feature_count": len(FEATURE_NAMES),
        "train": train_summary,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

