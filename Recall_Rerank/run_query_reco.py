from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Recall+Rerank (Non-LLM) query-based real-time POI recommendation."
    )
    parser.add_argument("--query", required=True, help="Natural-language user request.")
    parser.add_argument("--local-time", default="", help="Local time string. Default: now.")
    parser.add_argument("--lat", type=float, default=None, help="Current latitude.")
    parser.add_argument("--lon", type=float, default=None, help="Current longitude.")
    parser.add_argument("--city", default="", help="City hint for filtering/ranking.")
    parser.add_argument("--state", default="", help="State hint for filtering/ranking.")
    parser.add_argument("--user-id", default="anonymous", help="User ID.")
    parser.add_argument(
        "--long-term-notes",
        default="",
        help="Optional long-term preference text.",
    )
    parser.add_argument(
        "--recent-activity-notes",
        default="",
        help="Optional recent behavior text.",
    )
    parser.add_argument("--top-k", type=int, default=0, help="Final recommendation top-k.")
    parser.add_argument(
        "--model-path",
        default="Recall_Rerank/models/listwise_ce_model.json",
        help="Path to trained listwise CE model JSON.",
    )
    parser.add_argument(
        "--rerank-pool-size",
        type=int,
        default=0,
        help="Optional rerank pool size override.",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()

    from Recall_Rerank.pipeline import RecallRerankRealtimeRecommender
    from src.config import load_settings
    from src.schemas import GeoPoint, UserQueryContext

    local_time = args.local_time.strip() or datetime.now().isoformat(timespec="minutes")
    location = None
    if args.lat is not None and args.lon is not None:
        location = GeoPoint(lat=args.lat, lon=args.lon)

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
    settings = load_settings()
    recommender = RecallRerankRealtimeRecommender(
        settings=settings,
        model_path=args.model_path.strip(),
        rerank_pool_size=(args.rerank_pool_size if args.rerank_pool_size > 0 else None),
    )
    result = recommender.recommend(context=context, top_k=args.top_k or settings.final_top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

