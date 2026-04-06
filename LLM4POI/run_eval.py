from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM4POI (API-LLM variant) on processed eval data."
    )
    parser.add_argument(
        "--eval-queries",
        default="data/eval/yelp-indianapolis-eval-queries.jsonl",
        help="Path to eval query file.",
    )
    parser.add_argument(
        "--eval-candidates",
        default="data/eval/yelp-indianapolis-eval-candidates.jsonl",
        help="Path to eval candidate file.",
    )
    parser.add_argument(
        "--mode",
        choices=["constrained", "full"],
        default="constrained",
        help="Use candidate-constrained retrieval or full-corpus retrieval.",
    )
    parser.add_argument("--max-queries", type=int, default=0, help="0 means all.")
    parser.add_argument("--k-values", default="1,5,10", help="Comma-separated cutoffs.")
    parser.add_argument(
        "--variant",
        choices=["llm4poi", "llm4poi_star", "llm4poi_star2"],
        default="llm4poi",
        help="LLM4POI variants from paper: full, no-history, or user-history-only.",
    )
    parser.add_argument("--trajectory-gap-hours", type=int, default=24)
    parser.add_argument("--history-top-trajectories", type=int, default=20)
    parser.add_argument("--max-history-records", type=int, default=300)
    parser.add_argument("--max-current-records", type=int, default=50)
    parser.add_argument("--max-similarity-pool", type=int, default=1200)
    parser.add_argument("--max-query-trajectories", type=int, default=6000)
    parser.add_argument("--rerank-pool-size", type=int, default=0)
    parser.add_argument("--few-shot-examples", type=int, default=3)
    parser.add_argument("--llm-stage1-temperature", type=float, default=0.0)
    parser.add_argument("--llm-stage2-temperature", type=float, default=0.1)
    parser.add_argument("--llm-retrieval-blend", type=float, default=0.25)
    parser.add_argument(
        "--save-predictions",
        default="",
        help="Optional output path for per-query predictions JSONL.",
    )
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
        candidate_ids = row.get("candidate_business_ids", [])
        if not qid or not isinstance(candidate_ids, list):
            continue
        mapping[qid] = [str(x).strip() for x in candidate_ids if str(x).strip()]
    return mapping


def _parse_ks(raw: str) -> list[int]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    ks = sorted({int(x) for x in parts if int(x) > 0})
    return ks or [1, 5, 10]


def _dcg_from_rank(rank_1based: int) -> float:
    import math

    return 1.0 / math.log2(rank_1based + 1.0)


def _metric_at_k(ranked_ids: list[str], gt_id: str, k: int) -> dict[str, float]:
    topk = ranked_ids[:k]
    if gt_id not in topk:
        return {"hit": 0.0, "recall": 0.0, "ndcg": 0.0, "mrr": 0.0}
    rank = topk.index(gt_id) + 1
    return {
        "hit": 1.0,
        "recall": 1.0,
        "ndcg": _dcg_from_rank(rank),
        "mrr": 1.0 / rank,
    }


def _agg_init(ks: list[int]) -> dict[str, Any]:
    return {
        "count": 0,
        "metrics": {
            str(k): {"hit": 0.0, "recall": 0.0, "ndcg": 0.0, "mrr": 0.0}
            for k in ks
        },
    }


def _agg_update(agg: dict[str, Any], ranked_ids: list[str], gt_id: str, ks: list[int]) -> None:
    agg["count"] += 1
    for k in ks:
        mk = _metric_at_k(ranked_ids, gt_id, k)
        bucket = agg["metrics"][str(k)]
        bucket["hit"] += mk["hit"]
        bucket["recall"] += mk["recall"]
        bucket["ndcg"] += mk["ndcg"]
        bucket["mrr"] += mk["mrr"]


def _agg_finalize(agg: dict[str, Any]) -> dict[str, Any]:
    count = max(1, agg["count"])
    out = {"count": agg["count"], "metrics": {}}
    for k, bucket in agg["metrics"].items():
        out["metrics"][k] = {
            "hit": round(bucket["hit"] / count, 6),
            "recall": round(bucket["recall"] / count, 6),
            "ndcg": round(bucket["ndcg"] / count, 6),
            "mrr": round(bucket["mrr"] / count, 6),
        }
    return out


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


def _estimate_eval_total(path: Path, max_queries: int) -> int:
    total = 0
    for row in _iter_jsonl(path):
        query_id = str(row.get("query_id", "")).strip()
        gt_id = str((row.get("ground_truth") or {}).get("business_id", "")).strip()
        if not query_id or not gt_id:
            continue
        total += 1
        if max_queries > 0 and total >= max_queries:
            break
    return total


def _stream_progress(processed: int, total: int, start_ts: float) -> None:
    total = max(1, total)
    width = 24
    ratio = min(1.0, processed / total)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(1e-6, time.time() - start_ts)
    rate = processed / elapsed if processed > 0 else 0.0
    if processed > 0 and rate > 0.0:
        eta_sec = max(0.0, (total - processed) / rate)
        eta_text = f"{int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}"
    else:
        eta_text = "--:--"
    msg = (
        f"\rprogress [{bar}] {processed}/{total} "
        f"({ratio * 100:5.1f}%) | {rate:5.2f} q/s | ETA {eta_text}"
    )
    print(msg, end="", file=sys.stderr, flush=True)


def main() -> None:
    args = parse_args()
    from LLM4POI.pipeline import LLM4POIRealtimeRecommender
    from src.config import load_settings

    eval_queries_path = Path(args.eval_queries)
    eval_candidates_path = Path(args.eval_candidates)
    if not eval_queries_path.exists():
        raise FileNotFoundError(f"Eval queries not found: {eval_queries_path}")
    if args.mode == "constrained" and not eval_candidates_path.exists():
        raise FileNotFoundError(f"Eval candidates not found: {eval_candidates_path}")

    ks = _parse_ks(args.k_values)
    settings = load_settings()
    recommender = LLM4POIRealtimeRecommender(
        settings=settings,
        variant=args.variant,
        trajectory_gap_hours=args.trajectory_gap_hours,
        history_top_trajectories=args.history_top_trajectories,
        max_history_records=args.max_history_records,
        max_current_records=args.max_current_records,
        max_similarity_pool=args.max_similarity_pool,
        max_query_trajectories=args.max_query_trajectories,
        rerank_pool_size=(args.rerank_pool_size if args.rerank_pool_size > 0 else None),
        few_shot_examples=args.few_shot_examples,
        llm_stage1_temperature=args.llm_stage1_temperature,
        llm_stage2_temperature=args.llm_stage2_temperature,
        llm_retrieval_blend=args.llm_retrieval_blend,
    )
    candidate_map = (
        _load_candidate_map(eval_candidates_path) if args.mode == "constrained" else {}
    )

    global_agg = _agg_init(ks)
    by_support: dict[str, dict[str, Any]] = defaultdict(lambda: _agg_init(ks))
    pred_writer = None
    if args.save_predictions.strip():
        pred_path = Path(args.save_predictions.strip())
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_writer = pred_path.open("w", encoding="utf-8")

    total = _estimate_eval_total(eval_queries_path, args.max_queries)
    started = time.time()
    if total > 0:
        _stream_progress(0, total, started)

    processed = 0
    for row in _iter_jsonl(eval_queries_path):
        query_id = str(row.get("query_id", "")).strip()
        gt = row.get("ground_truth") or {}
        gt_id = str(gt.get("business_id", "")).strip()
        if not query_id or not gt_id:
            continue

        context = _to_context(row)
        if args.mode == "constrained":
            result = recommender.recommend_with_candidates(
                context=context,
                candidate_business_ids=candidate_map.get(query_id, []),
                top_k=max(ks),
            )
        else:
            result = recommender.recommend(context=context, top_k=max(ks))

        ranked_ids = [
            str(item.get("business", {}).get("business_id", "")).strip()
            for item in result.get("recommendations", [])
            if str(item.get("business", {}).get("business_id", "")).strip()
        ]
        support = str(
            (row.get("evaluation_slice") or {}).get("support_level_after_holdout", "unknown")
        )
        _agg_update(global_agg, ranked_ids, gt_id, ks)
        _agg_update(by_support[support], ranked_ids, gt_id, ks)

        if pred_writer:
            pred_writer.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "user_id": row.get("user_id", ""),
                        "gt_business_id": gt_id,
                        "ranked_business_ids": ranked_ids,
                        "mode": args.mode,
                        "variant": args.variant,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        processed += 1
        if total > 0:
            _stream_progress(processed, total, started)
        if args.max_queries > 0 and processed >= args.max_queries:
            break

    if total > 0:
        print("", file=sys.stderr, flush=True)
    if pred_writer:
        pred_writer.close()

    summary = {
        "mode": args.mode,
        "variant": args.variant,
        "eval_queries": str(eval_queries_path),
        "eval_candidates": str(eval_candidates_path) if args.mode == "constrained" else "",
        "processed_queries": processed,
        "k_values": ks,
        "overall": _agg_finalize(global_agg),
        "by_support_level": {
            support: _agg_finalize(agg) for support, agg in sorted(by_support.items())
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
