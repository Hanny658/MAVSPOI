from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

from src.agents import (
    AggregatorAgent,
    RouterAgent,
    VotingOutput,
    build_default_registry,
)
from src.config import Settings, load_mavspoi_config
from src.openai_client import OpenAIService
from src.profile_loader import load_user_profiles
from src.retrieval import CandidateRetriever
from src.schemas import CandidateScore, UserQueryContext
from src.yelp_loader import load_yelp_businesses


def _profile_top_categories(profile: dict[str, Any], limit: int = 5) -> list[str]:
    category_rows = profile.get("category_pref", {}).get("top_categories", [])
    if not isinstance(category_rows, list):
        return []
    output: list[str] = []
    for row in category_rows:
        if not isinstance(row, dict):
            continue
        cat = str(row.get("category", "")).strip()
        if cat:
            output.append(cat)
        if len(output) >= limit:
            break
    return output


def _compact_profile_for_agents(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {}
    support = profile.get("support", {})
    static = profile.get("static", {})
    temporal_pref = profile.get("temporal_pref", {})
    price_pref = profile.get("price_pref", {})
    spatial_pref = profile.get("spatial_pref", {})

    return {
        "profile_version": profile.get("profile_version", ""),
        "support_level": support.get("support_level", "unknown"),
        "interaction_evidence_count": support.get("interaction_evidence_count", 0),
        "top_categories": _profile_top_categories(profile, limit=8),
        "dominant_price_level": price_pref.get("dominant_price_level", ""),
        "active_hours_top3": temporal_pref.get("active_hours_top3", []),
        "weekend_ratio": temporal_pref.get("weekend_ratio", 0.0),
        "radius_km_p50": spatial_pref.get("radius_km_p50", None),
        "radius_km_p90": spatial_pref.get("radius_km_p90", None),
        "user_average_stars": static.get("user_average_stars", 0.0),
        "fans": static.get("fans", 0),
        "friend_count": static.get("friend_count", 0),
    }


def _profile_to_long_term_notes(profile: dict[str, Any] | None) -> str:
    if not profile:
        return ""
    support = profile.get("support", {})
    spatial = profile.get("spatial_pref", {})
    business_quality = profile.get("business_quality_pref", {})
    price_pref = profile.get("price_pref", {})

    pieces: list[str] = []
    pieces.append(f"support_level={support.get('support_level', 'unknown')}")
    pieces.append(f"interaction_count={support.get('interaction_evidence_count', 0)}")

    cats = _profile_top_categories(profile, limit=5)
    if cats:
        pieces.append("top_categories=" + ", ".join(cats))

    dominant_price = str(price_pref.get("dominant_price_level", "")).strip()
    if dominant_price:
        pieces.append(f"price_level={dominant_price}")

    if spatial.get("radius_km_p50") is not None:
        pieces.append(f"radius_p50_km={spatial.get('radius_km_p50')}")
    if spatial.get("radius_km_p90") is not None:
        pieces.append(f"radius_p90_km={spatial.get('radius_km_p90')}")
    if business_quality.get("avg_business_stars") is not None:
        pieces.append(
            f"avg_visited_business_stars={business_quality.get('avg_business_stars')}"
        )
    return "Profile-derived long-term signals: " + "; ".join(pieces)


def _profile_to_recent_notes(profile: dict[str, Any] | None) -> str:
    if not profile:
        return ""
    temporal = profile.get("temporal_pref", {})
    checkin = profile.get("checkin_context_pref", {})
    pieces: list[str] = []
    active_hours = temporal.get("active_hours_top3", [])
    if isinstance(active_hours, list) and active_hours:
        pieces.append("active_hours_top3=" + ",".join(str(x) for x in active_hours))
    weekend_ratio = temporal.get("weekend_ratio")
    if weekend_ratio is not None:
        pieces.append(f"weekend_ratio={weekend_ratio}")
    avg_checkins = checkin.get("avg_business_checkins")
    if avg_checkins is not None:
        pieces.append(f"avg_business_checkins={avg_checkins}")
    if not pieces:
        return ""
    return "Profile-derived short-term hints: " + "; ".join(pieces)


def _merge_notes(manual_note: str, profile_note: str) -> str:
    manual_note = manual_note.strip()
    profile_note = profile_note.strip()
    if manual_note and profile_note:
        return manual_note + " | " + profile_note
    return manual_note or profile_note


class MAVSPOIRealtimeRecommender:
    def __init__(self, settings: Settings, config_path: str | None = None) -> None:
        self.settings = settings
        self.mav_config = load_mavspoi_config(config_path=config_path)

        self.openai_service = OpenAIService(settings)
        self.businesses = load_yelp_businesses(
            business_json_path=settings.yelp_business_json,
            max_businesses=settings.yelp_max_businesses,
            city_filter=settings.yelp_city_filter,
        )
        self.user_profiles = load_user_profiles(settings.yelp_profile_json)
        self.retriever = CandidateRetriever(
            openai_service=self.openai_service,
            businesses=self.businesses,
            embed_cache_path=settings.embed_cache_path,
        )

        self.registry = build_default_registry(
            llm=self.openai_service,
            voting_config=self.mav_config.get("voting", {}),
        )
        self.router = RouterAgent(
            llm=self.openai_service,
            registry=self.registry,
            config=self.mav_config.get("router", {}),
            base_weights=self.mav_config.get("aggregator", {}).get("weights", {}),
        )
        self.aggregator = AggregatorAgent(self.mav_config.get("aggregator", {}))

    def _build_enriched_context(self, context: UserQueryContext) -> tuple[UserQueryContext, dict[str, Any]]:
        profile = self.user_profiles.get(context.user_id)
        profile_long = _profile_to_long_term_notes(profile)
        profile_recent = _profile_to_recent_notes(profile)
        enriched = UserQueryContext(
            query_text=context.query_text,
            local_time=context.local_time,
            location=context.location,
            city=context.city,
            state=context.state,
            user_id=context.user_id,
            long_term_notes=_merge_notes(context.long_term_notes, profile_long),
            recent_activity_notes=_merge_notes(context.recent_activity_notes, profile_recent),
        )
        return enriched, _compact_profile_for_agents(profile)

    def _apply_global_constraints(
        self,
        candidates: list[CandidateScore],
        constraints: dict[str, Any],
    ) -> list[CandidateScore]:
        city = str(constraints.get("city", "")).strip().lower()
        state = str(constraints.get("state", "")).strip().lower()
        open_now = bool(constraints.get("open_now", False))
        max_distance = float(constraints.get("max_distance_km", 0.0) or 0.0)

        output: list[CandidateScore] = []
        for c in candidates:
            if city and c.business.city.strip().lower() != city:
                continue
            if state and c.business.state.strip().lower() != state:
                continue
            if open_now and int(c.business.is_open) != 1:
                continue
            if max_distance > 0 and c.distance_km is not None and float(c.distance_km) > max_distance:
                continue
            output.append(c)
        return output or candidates

    def _run_voting(
        self,
        context: UserQueryContext,
        candidates: list[CandidateScore],
        profile_features: dict[str, Any],
        activated_agent_ids: list[str],
    ) -> dict[str, VotingOutput]:
        out: dict[str, VotingOutput] = {}
        worker_count = int(
            self.mav_config.get("voting", {}).get("parallel_workers", len(activated_agent_ids))
        )
        worker_count = max(1, min(worker_count, max(1, len(activated_agent_ids))))

        def _score(agent_id: str):
            agent = self.registry.get(agent_id)
            if agent is None:
                return agent_id, None
            try:
                result = agent.score_candidates(
                    context=context,
                    candidates=candidates,
                    profile_features=profile_features,
                )
                return agent_id, result
            except Exception:
                return agent_id, None

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_score, aid) for aid in activated_agent_ids]
            for fut in as_completed(futures):
                aid, result = fut.result()
                if result is not None:
                    out[aid] = result
        return out

    def _finalize_recommendations(
        self,
        candidate_pool: list[CandidateScore],
        rows: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        by_id = {c.business.business_id: c for c in candidate_pool}
        output: list[dict[str, Any]] = []
        for row in rows:
            business_id = str(row.get("business_id", "")).strip()
            candidate = by_id.get(business_id)
            if candidate is None:
                continue
            output.append(
                {
                    "business": candidate.business.to_compact_dict(),
                    "ranking_score": float(row.get("final_score", 0.0)),
                    "reason": str(row.get("reason", "")),
                    "fit_tags": row.get("fit_tags", []),
                    "contribution": row.get("contribution", {}),
                    "retrieval_components": {
                        "retrieval_score": candidate.score,
                        "text_similarity": candidate.text_similarity,
                        "geo_score": candidate.geo_score,
                        "popularity_score": candidate.popularity_score,
                        "distance_km": candidate.distance_km,
                    },
                }
            )
            if len(output) >= top_k:
                break
        return output

    def recommend(self, context: UserQueryContext, top_k: int | None = None) -> dict[str, Any]:
        return self._recommend_impl(context=context, top_k=top_k, candidate_business_ids=None)

    def recommend_with_candidates(
        self,
        context: UserQueryContext,
        candidate_business_ids: list[str],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        return self._recommend_impl(
            context=context,
            top_k=top_k,
            candidate_business_ids=candidate_business_ids,
        )

    def _recommend_impl(
        self,
        context: UserQueryContext,
        top_k: int | None,
        candidate_business_ids: list[str] | None,
    ) -> dict[str, Any]:
        top_k = top_k or self.settings.final_top_k
        enriched_context, profile_features = self._build_enriched_context(context)

        if candidate_business_ids:
            initial_candidates = self.retriever.retrieve_from_pool(
                context=enriched_context,
                candidate_business_ids=candidate_business_ids,
                top_k=self.settings.retrieval_top_k,
            )
            retrieval_mode = "candidate_constrained"
        else:
            initial_candidates = self.retriever.retrieve(
                context=enriched_context,
                top_k=self.settings.retrieval_top_k,
            )
            retrieval_mode = "full_corpus"

        voting_pool_size = int(
            self.mav_config.get("voting", {}).get("candidate_pool_size", max(top_k, self.settings.forecaster_top_k))
        )
        voting_pool_size = max(top_k, voting_pool_size)
        candidate_pool = initial_candidates[:voting_pool_size]

        router_decision = self.router.run(
            context=enriched_context,
            profile_features=profile_features,
            candidates=candidate_pool,
        )
        constrained_pool = self._apply_global_constraints(
            candidates=candidate_pool,
            constraints=router_decision.global_constraints,
        )
        activated_ids = [a.agent_id for a in router_decision.activated_agents]
        votes = self._run_voting(
            context=enriched_context,
            candidates=constrained_pool,
            profile_features=profile_features,
            activated_agent_ids=activated_ids,
        )
        aggregation = self.aggregator.run(
            router_decision=router_decision,
            votes=votes,
            candidates=constrained_pool,
            top_k=top_k,
        )
        recommendation_rows = [
            {
                "business_id": row.business_id,
                "final_score": row.final_score,
                "reason": row.reason,
                "fit_tags": row.fit_tags,
                "contribution": row.contribution,
            }
            for row in aggregation.recommendations
        ]
        recommendations = self._finalize_recommendations(
            candidate_pool=constrained_pool,
            rows=recommendation_rows,
            top_k=top_k,
        )

        return {
            "task": "query_based_real_time_poi_recommendation",
            "system": "mavspoi_modular_v0",
            "context": context.to_dict(),
            "enriched_context": enriched_context.to_dict(),
            "retrieval_mode": retrieval_mode,
            "user_profile_features": profile_features,
            "intermediate": {
                "retrieval_pool_size": len(initial_candidates),
                "voting_pool_size": len(candidate_pool),
                "post_constraint_pool_size": len(constrained_pool),
                "router_decision": asdict(router_decision),
                "activated_agent_count": len(activated_ids),
                "agent_status": {aid: votes[aid].status for aid in votes},
            },
            "recommendations": recommendations,
            "summary": aggregation.summary,
        }
