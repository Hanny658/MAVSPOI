from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-query recommendation using LLM4POI (API-LLM variant)."
    )
    parser.add_argument("--query", required=True, help="Natural language query.")
    parser.add_argument("--local-time", default="", help="Local time string. Default: now.")
    parser.add_argument("--lat", type=float, default=None, help="Current latitude.")
    parser.add_argument("--lon", type=float, default=None, help="Current longitude.")
    parser.add_argument("--city", default="", help="City hint.")
    parser.add_argument("--state", default="", help="State hint.")
    parser.add_argument("--user-id", default="anonymous", help="User ID.")
    parser.add_argument("--long-term-notes", default="", help="Optional long-term notes.")
    parser.add_argument("--recent-activity-notes", default="", help="Optional recent notes.")
    parser.add_argument("--top-k", type=int, default=0, help="Final top-k.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from LLM4POI.pipeline import LLM4POIRealtimeRecommender
    from src.config import load_settings
    from src.schemas import GeoPoint, UserQueryContext

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
        user_id=args.user_id.strip() or "anonymous",
        long_term_notes=args.long_term_notes.strip(),
        recent_activity_notes=args.recent_activity_notes.strip(),
    )
    top_k = args.top_k if args.top_k > 0 else settings.final_top_k
    result = recommender.recommend(context=context, top_k=top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
