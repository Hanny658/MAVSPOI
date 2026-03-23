from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from CoMaPOI_styled import prompts
from src.openai_client import OpenAIService
from src.schemas import CandidateScore, UserQueryContext


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _sanitize_ids(ids: list[Any], allowed: set[str], top_k: int) -> list[str]:
    out: list[str] = []
    for item in ids:
        value = str(item).strip()
        if value in allowed:
            out.append(value)
    return _dedup_keep_order(out)[:top_k]


@dataclass
class ProfilerOutput:
    long_term_profile: dict[str, Any]
    short_term_pattern: dict[str, Any]
    hard_constraints: dict[str, Any]


@dataclass
class ForecasterOutput:
    long_term_candidate_ids: list[str]
    short_term_candidate_ids: list[str]
    merged_candidate_ids: list[str]
    rerank_rationale: list[str]


@dataclass
class PredictorOutput:
    recommendations: list[dict[str, Any]]
    final_summary: str


class ProfilerAgent:
    def __init__(self, llm: OpenAIService) -> None:
        self.llm = llm

    def run(
        self,
        context: UserQueryContext,
        initial_candidates: list[CandidateScore],
        profile_features: dict[str, Any] | None = None,
        profiler_tools: dict[str, Any] | None = None,
    ) -> ProfilerOutput:
        payload = {
            "task": "query_based_real_time_poi_recommendation",
            "context": context.to_dict(),
            "user_profile_features": profile_features or {},
            "trajectory_tools": profiler_tools or {},
            "retrieval_snapshot": [
                c.to_compact_dict() for c in initial_candidates[:15]
            ],
        }
        try:
            result = self.llm.chat_json(
                system_prompt=prompts.profiler_system_prompt(),
                user_prompt=prompts.profiler_user_prompt(payload),
                temperature=0.1,
            )
        except Exception:
            result = {}

        long_term = result.get("long_term_profile", {}) if isinstance(result, dict) else {}
        short_term = result.get("short_term_pattern", {}) if isinstance(result, dict) else {}
        constraints = result.get("hard_constraints", {}) if isinstance(result, dict) else {}

        if not long_term:
            long_term = {
                "summary": "Preference inferred mainly from current query.",
                "preferred_categories": [],
                "price_preference": "unknown",
                "distance_preference_km": 8,
                "avoid_categories": [],
            }
        if not short_term:
            short_term = {
                "intent_summary": context.query_text,
                "time_sensitivity": "high",
                "must_have": [],
                "nice_to_have": [],
                "avoid_now": [],
            }
        if not constraints:
            constraints = {
                "city": context.city,
                "open_now": True,
                "max_distance_km": 10,
            }

        return ProfilerOutput(
            long_term_profile=long_term,
            short_term_pattern=short_term,
            hard_constraints=constraints,
        )


class ForecasterAgent:
    def __init__(self, llm: OpenAIService) -> None:
        self.llm = llm

    def run(
        self,
        context: UserQueryContext,
        profiler_output: ProfilerOutput,
        initial_candidates: list[CandidateScore],
        short_term_initial_candidate_ids: list[str] | None = None,
        top_k: int = 25,
        profile_features: dict[str, Any] | None = None,
    ) -> ForecasterOutput:
        allowed_ids = {c.business.business_id for c in initial_candidates}
        short_term_initial_candidate_ids = short_term_initial_candidate_ids or []
        payload = {
            "context": context.to_dict(),
            "user_profile_features": profile_features or {},
            "profiler_output": {
                "long_term_profile": profiler_output.long_term_profile,
                "short_term_pattern": profiler_output.short_term_pattern,
                "hard_constraints": profiler_output.hard_constraints,
            },
            "initial_candidates": [c.to_compact_dict() for c in initial_candidates],
            "short_term_initial_candidate_ids": [
                x for x in short_term_initial_candidate_ids if x in allowed_ids
            ][: max(10, top_k * 2)],
            "target_size_each_list": top_k,
        }
        try:
            result = self.llm.chat_json(
                system_prompt=prompts.forecaster_system_prompt(),
                user_prompt=prompts.forecaster_user_prompt(payload),
                temperature=0.1,
            )
        except Exception:
            result = {}

        long_ids = _sanitize_ids(
            result.get("long_term_candidate_ids", []), allowed_ids, top_k
        )
        short_ids = _sanitize_ids(
            result.get("short_term_candidate_ids", []), allowed_ids, top_k
        )
        merged_ids = _sanitize_ids(
            result.get("merged_candidate_ids", []), allowed_ids, top_k
        )

        if not long_ids:
            long_ids = [c.business.business_id for c in initial_candidates[:top_k]]
        if not short_ids:
            if short_term_initial_candidate_ids:
                short_ids = _sanitize_ids(
                    short_term_initial_candidate_ids,
                    allowed_ids,
                    top_k,
                )
            else:
                short_ids = [c.business.business_id for c in initial_candidates[:top_k]]
        if not merged_ids:
            merged_ids = _dedup_keep_order(long_ids + short_ids)[:top_k]

        rationale = result.get("rerank_rationale", [])
        if not isinstance(rationale, list):
            rationale = []
        rationale = [str(item) for item in rationale][:8]

        return ForecasterOutput(
            long_term_candidate_ids=long_ids,
            short_term_candidate_ids=short_ids,
            merged_candidate_ids=merged_ids,
            rerank_rationale=rationale,
        )


class PredictorAgent:
    def __init__(self, llm: OpenAIService) -> None:
        self.llm = llm

    def run(
        self,
        context: UserQueryContext,
        profiler_output: ProfilerOutput,
        forecaster_output: ForecasterOutput,
        final_candidates: list[CandidateScore],
        candidate_universe: list[CandidateScore] | None = None,
        top_k: int = 10,
        profile_features: dict[str, Any] | None = None,
        allow_out_of_set: bool = True,
        out_of_set_penalty: float = 0.85,
    ) -> PredictorOutput:
        allowed_ids = {c.business.business_id for c in final_candidates}
        candidate_universe = candidate_universe or final_candidates
        universe_ids = {c.business.business_id for c in candidate_universe}
        payload = {
            "context": context.to_dict(),
            "user_profile_features": profile_features or {},
            "profiler_output": {
                "long_term_profile": profiler_output.long_term_profile,
                "short_term_pattern": profiler_output.short_term_pattern,
            },
            "forecaster_output": {
                "long_term_candidate_ids": forecaster_output.long_term_candidate_ids,
                "short_term_candidate_ids": forecaster_output.short_term_candidate_ids,
                "merged_candidate_ids": forecaster_output.merged_candidate_ids,
            },
            "final_candidates": [c.to_compact_dict() for c in final_candidates],
            "candidate_universe": [c.to_compact_dict() for c in candidate_universe[:60]],
            "allow_out_of_set": bool(allow_out_of_set),
            "top_k": top_k,
        }
        try:
            result = self.llm.chat_json(
                system_prompt=prompts.predictor_system_prompt(),
                user_prompt=prompts.predictor_user_prompt(payload),
                temperature=0.1,
            )
        except Exception:
            result = {}

        recs = result.get("recommendations", [])
        if not isinstance(recs, list):
            recs = []

        sanitized_recs: list[dict[str, Any]] = []
        out_of_set_count = 0
        seen: set[str] = set()
        for row in recs:
            if not isinstance(row, dict):
                continue
            business_id = str(row.get("business_id", "")).strip()
            is_out_of_set = business_id not in allowed_ids
            if is_out_of_set and (
                not allow_out_of_set
                or business_id not in universe_ids
                or out_of_set_count >= 2
            ):
                continue
            if business_id in seen:
                continue
            try:
                score = float(row.get("score", 0.5))
            except (TypeError, ValueError):
                score = 0.5
            if is_out_of_set:
                score = score * max(0.0, min(1.0, out_of_set_penalty))
                out_of_set_count += 1
            reason = str(row.get("reason", "")).strip()
            fit_tags_raw = row.get("fit_tags", [])
            fit_tags = [str(x) for x in fit_tags_raw] if isinstance(fit_tags_raw, list) else []
            if is_out_of_set:
                fit_tags.append("out_of_set_candidate")
            sanitized_recs.append(
                {
                    "business_id": business_id,
                    "score": max(0.0, min(1.0, score)),
                    "reason": reason,
                    "fit_tags": fit_tags[:6],
                }
            )
            seen.add(business_id)
            if len(sanitized_recs) >= top_k:
                break

        if not sanitized_recs:
            for c in final_candidates[:top_k]:
                sanitized_recs.append(
                    {
                        "business_id": c.business.business_id,
                        "score": max(0.0, min(1.0, c.score)),
                        "reason": "Fallback to retriever ranking.",
                        "fit_tags": ["retrieval_fallback"],
                    }
                )

        final_summary = result.get("final_summary", "")
        if not isinstance(final_summary, str) or not final_summary.strip():
            final_summary = "Final ranking based on integrated profile, intent, and candidate refinement."

        return PredictorOutput(
            recommendations=sanitized_recs,
            final_summary=final_summary,
        )
