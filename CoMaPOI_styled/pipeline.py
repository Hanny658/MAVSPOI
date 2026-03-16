from __future__ import annotations

from dataclasses import asdict
from typing import Any

from CoMaPOI_styled.agents import (
    ForecasterAgent,
    ForecasterOutput,
    PredictorAgent,
    PredictorOutput,
    ProfilerAgent,
    ProfilerOutput,
)
from src.config import Settings
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
    category_pref = profile.get("category_pref", {})
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
    pieces.append(
        f"interaction_count={support.get('interaction_evidence_count', 0)}"
    )

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


class CoMaPOIStyledRealtimeRecommender:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
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
        self.profiler = ProfilerAgent(self.openai_service)
        self.forecaster = ForecasterAgent(self.openai_service)
        self.predictor = PredictorAgent(self.openai_service)

    def _slice_final_candidates(
        self,
        initial: list[CandidateScore],
        forecast: ForecasterOutput,
    ) -> list[CandidateScore]:
        by_id = {c.business.business_id: c for c in initial}
        merged = []
        for business_id in forecast.merged_candidate_ids:
            if business_id in by_id:
                merged.append(by_id[business_id])
        if not merged:
            merged = initial[: self.settings.forecaster_top_k]
        return merged

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

    def recommend(self, context: UserQueryContext, top_k: int | None = None) -> dict[str, Any]:
        return self._recommend_impl(
            context=context,
            top_k=top_k,
            candidate_business_ids=None,
        )

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
        profiler_output: ProfilerOutput = self.profiler.run(
            enriched_context,
            initial_candidates,
            profile_features=profile_features,
        )
        forecaster_output: ForecasterOutput = self.forecaster.run(
            context=enriched_context,
            profiler_output=profiler_output,
            initial_candidates=initial_candidates,
            top_k=self.settings.forecaster_top_k,
            profile_features=profile_features,
        )
        final_candidates = self._slice_final_candidates(initial_candidates, forecaster_output)
        predictor_output: PredictorOutput = self.predictor.run(
            context=enriched_context,
            profiler_output=profiler_output,
            forecaster_output=forecaster_output,
            final_candidates=final_candidates,
            top_k=top_k,
            profile_features=profile_features,
        )

        final_by_id = {c.business.business_id: c for c in final_candidates}
        recommendations: list[dict[str, Any]] = []
        for row in predictor_output.recommendations:
            business_id = row["business_id"]
            candidate = final_by_id.get(business_id)
            if candidate is None:
                continue
            merged = {
                "business": candidate.business.to_compact_dict(),
                "ranking_score": row["score"],
                "reason": row["reason"],
                "fit_tags": row["fit_tags"],
                "retrieval_components": {
                    "retrieval_score": candidate.score,
                    "text_similarity": candidate.text_similarity,
                    "geo_score": candidate.geo_score,
                    "popularity_score": candidate.popularity_score,
                    "distance_km": candidate.distance_km,
                },
            }
            recommendations.append(merged)

        return {
            "task": "query_based_real_time_poi_recommendation",
            "context": context.to_dict(),
            "enriched_context": enriched_context.to_dict(),
            "retrieval_mode": retrieval_mode,
            "user_profile_features": profile_features,
            "intermediate": {
                "profiler_output": asdict(profiler_output),
                "forecaster_output": asdict(forecaster_output),
            },
            "recommendations": recommendations[:top_k],
            "summary": predictor_output.final_summary,
        }
