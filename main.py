from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MAVSPOI main entry. Use subcommands: query / eval."
    )
    parser.add_argument(
        "--config-yaml",
        default="",
        help="Optional config YAML path. Defaults to CONFIG_YAML_PATH or config.yaml.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    query_parser = sub.add_parser("query", help="Run a single query recommendation.")
    query_parser.add_argument("--query", required=True, help="Natural-language user request.")
    query_parser.add_argument("--local-time", default="", help="Local time string. Default: now.")
    query_parser.add_argument("--lat", type=float, default=None, help="Current latitude.")
    query_parser.add_argument("--lon", type=float, default=None, help="Current longitude.")
    query_parser.add_argument("--city", default="", help="City hint.")
    query_parser.add_argument("--state", default="", help="State hint.")
    query_parser.add_argument("--user-id", default="anonymous", help="User ID.")
    query_parser.add_argument("--long-term-notes", default="", help="Optional long-term notes.")
    query_parser.add_argument("--recent-activity-notes", default="", help="Optional recent behavior notes.")
    query_parser.add_argument("--top-k", type=int, default=0, help="Final top-k.")

    eval_parser = sub.add_parser("eval", help="Run batch evaluation.")
    eval_parser.add_argument(
        "--eval-queries",
        default="data/eval/yelp-indianapolis-eval-queries.jsonl",
        help="Path to eval query file.",
    )
    eval_parser.add_argument(
        "--eval-candidates",
        default="data/eval/yelp-indianapolis-eval-candidates.jsonl",
        help="Path to eval candidate file.",
    )
    eval_parser.add_argument(
        "--mode",
        choices=["constrained", "full"],
        default="constrained",
        help="Use candidate-constrained retrieval or full-corpus retrieval.",
    )
    eval_parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Maximum queries to evaluate. 0 means all.",
    )
    eval_parser.add_argument(
        "--k-values",
        default="1,5,10",
        help="Comma-separated cutoffs for ranking metrics.",
    )
    eval_parser.add_argument(
        "--save-predictions",
        default="",
        help="Optional output path for per-query predictions JSONL.",
    )
    return parser


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


def _dcg_from_gain(relevance: float, rank_1based: int) -> float:
    import math

    rel = max(0.0, float(relevance))
    return (2.0**rel - 1.0) / math.log2(rank_1based + 1.0)


def _extract_soft_targets(row: dict[str, Any]) -> dict[str, float]:
    mapping: dict[str, float] = {}
    soft = row.get("ground_truth_soft")
    if isinstance(soft, dict):
        items = soft.get("items", [])
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                bid = str(item.get("business_id", "")).strip()
                if not bid:
                    continue
                try:
                    rel = float(item.get("relevance", 0.0))
                except Exception:
                    continue
                if rel <= 0.0:
                    continue
                mapping[bid] = max(mapping.get(bid, 0.0), rel)
    gt_id = str((row.get("ground_truth") or {}).get("business_id", "")).strip()
    if gt_id and gt_id not in mapping:
        mapping[gt_id] = 1.0
    return mapping


def _hard_metric_at_k(ranked_ids: list[str], gt_id: str, k: int) -> dict[str, float]:
    topk = ranked_ids[:k]
    if gt_id not in topk:
        return {"hit": 0.0, "recall": 0.0, "ndcg": 0.0}
    rank = topk.index(gt_id) + 1
    return {
        "hit": 1.0,
        "recall": 1.0,
        "ndcg": _dcg_from_rank(rank),
    }


def _soft_metric_at_k(
    ranked_ids: list[str],
    soft_targets: dict[str, float],
    k: int,
) -> dict[str, float]:
    if not soft_targets:
        return {"ndcg_soft": 0.0, "wrecall_soft": 0.0}
    topk = ranked_ids[:k]
    dcg = 0.0
    seen: set[str] = set()
    covered_rel = 0.0
    for idx, bid in enumerate(topk, start=1):
        rel = float(soft_targets.get(bid, 0.0))
        if rel <= 0.0:
            continue
        dcg += _dcg_from_gain(rel, idx)
        if bid not in seen:
            covered_rel += rel
            seen.add(bid)
    ideal_rels = sorted((float(v) for v in soft_targets.values() if v > 0.0), reverse=True)[:k]
    idcg = sum(_dcg_from_gain(rel, idx) for idx, rel in enumerate(ideal_rels, start=1))
    total_rel = sum(float(v) for v in soft_targets.values() if v > 0.0)
    return {
        "ndcg_soft": (dcg / idcg) if idcg > 1e-12 else 0.0,
        "wrecall_soft": (covered_rel / total_rel) if total_rel > 1e-12 else 0.0,
    }


def _agg_init(ks: list[int]) -> dict[str, Any]:
    return {
        "count": 0,
        "metrics": {
            str(k): {
                "hit": 0.0,
                "recall": 0.0,
                "ndcg": 0.0,
                "ndcg_soft": 0.0,
                "wrecall_soft": 0.0,
            }
            for k in ks
        },
    }


def _agg_update(
    agg: dict[str, Any],
    ranked_ids: list[str],
    gt_id: str,
    soft_targets: dict[str, float],
    ks: list[int],
) -> None:
    agg["count"] += 1
    for k in ks:
        mk = _hard_metric_at_k(ranked_ids, gt_id, k)
        sk = _soft_metric_at_k(ranked_ids, soft_targets, k)
        bucket = agg["metrics"][str(k)]
        bucket["hit"] += mk["hit"]
        bucket["recall"] += mk["recall"]
        bucket["ndcg"] += mk["ndcg"]
        bucket["ndcg_soft"] += sk["ndcg_soft"]
        bucket["wrecall_soft"] += sk["wrecall_soft"]


def _agg_finalize(agg: dict[str, Any]) -> dict[str, Any]:
    count = max(1, agg["count"])
    out = {"count": agg["count"], "metrics": {}}
    for k, bucket in agg["metrics"].items():
        out["metrics"][k] = {
            "hit": round(bucket["hit"] / count, 6),
            "recall": round(bucket["recall"] / count, 6),
            "ndcg": round(bucket["ndcg"] / count, 6),
            "ndcg_soft": round(bucket["ndcg_soft"] / count, 6),
            "wrecall_soft": round(bucket["wrecall_soft"] / count, 6),
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
        soft_targets = _extract_soft_targets(row)
        if not query_id or not soft_targets:
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


def run_query(args: argparse.Namespace) -> None:
    from src.config import load_settings
    from src.mavspoi_pipeline import MAVSPOIRealtimeRecommender
    from src.schemas import GeoPoint, UserQueryContext

    settings = load_settings()
    recommender = MAVSPOIRealtimeRecommender(
        settings=settings,
        config_path=args.config_yaml.strip() or None,
    )

    local_time = args.local_time.strip() or datetime.now().isoformat(timespec="minutes")
    location = None
    if args.lat is not None and args.lon is not None:
        location = GeoPoint(lat=float(args.lat), lon=float(args.lon))

    context = UserQueryContext(
        query_text=args.query.strip(),
        local_time=local_time,
        location=location,
        city=args.city.strip(),
        state=args.state.strip(),
        user_id=args.user_id.strip(),
        long_term_notes=args.long_term_notes.strip(),
        recent_activity_notes=args.recent_activity_notes.strip(),
    )
    result = recommender.recommend(context=context, top_k=args.top_k or settings.final_top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_eval(args: argparse.Namespace) -> None:
    from src.config import load_settings
    from src.mavspoi_pipeline import MAVSPOIRealtimeRecommender

    settings = load_settings()
    recommender = MAVSPOIRealtimeRecommender(
        settings=settings,
        config_path=args.config_yaml.strip() or None,
    )

    eval_queries_path = Path(args.eval_queries)
    eval_candidates_path = Path(args.eval_candidates)
    if not eval_queries_path.exists():
        raise FileNotFoundError(f"Eval queries not found: {eval_queries_path}")
    if args.mode == "constrained" and not eval_candidates_path.exists():
        raise FileNotFoundError(f"Eval candidates not found: {eval_candidates_path}")

    ks = _parse_ks(args.k_values)
    candidate_map = _load_candidate_map(eval_candidates_path) if args.mode == "constrained" else {}
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
        soft_targets = _extract_soft_targets(row)
        gt = row.get("ground_truth") or {}
        gt_id = str(gt.get("business_id", "")).strip()
        if not query_id or not soft_targets or not gt_id:
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
        _agg_update(global_agg, ranked_ids, gt_id, soft_targets, ks)
        _agg_update(by_support[support], ranked_ids, gt_id, soft_targets, ks)

        if pred_writer:
            pred_writer.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "user_id": row.get("user_id", ""),
                        "gt_business_id": gt_id,
                        "ranked_business_ids": ranked_ids,
                        "mode": args.mode,
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "query":
        run_query(args)
    elif args.command == "eval":
        run_eval(args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
