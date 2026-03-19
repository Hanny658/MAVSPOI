from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from SingleAgent import prompts
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


@dataclass
class SingleAgentOutput:
    recommendations: list[dict[str, Any]]
    final_summary: str
    planning: dict[str, Any]


class SingleRecommenderAgent:
    def __init__(self, llm: OpenAIService, cot_mode: str = "off") -> None:
        if cot_mode not in {"off", "embedded", "two_pass"}:
            raise ValueError(f"Unsupported cot_mode: {cot_mode}")
        self.llm = llm
        self.cot_mode = cot_mode

    def _sanitize_recommendations(
        self,
        raw_rows: Any,
        allowed_ids: set[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_rows, list):
            return []

        sanitized: list[dict[str, Any]] = []
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            business_id = str(row.get("business_id", "")).strip()
            if business_id not in allowed_ids:
                continue

            try:
                score = float(row.get("score", 0.5))
            except (TypeError, ValueError):
                score = 0.5
            score = max(0.0, min(1.0, score))

            reason = str(row.get("reason", "")).strip()
            tags_raw = row.get("fit_tags", [])
            fit_tags = [str(x) for x in tags_raw] if isinstance(tags_raw, list) else []
            sanitized.append(
                {
                    "business_id": business_id,
                    "score": score,
                    "reason": reason,
                    "fit_tags": fit_tags[:6],
                }
            )
            if len(sanitized) >= top_k:
                break
        return sanitized

    def _fallback_recommendations(
        self,
        candidates: list[CandidateScore],
        top_k: int,
    ) -> list[dict[str, Any]]:
        return [
            {
                "business_id": c.business.business_id,
                "score": max(0.0, min(1.0, c.score)),
                "reason": "Fallback to retriever ranking.",
                "fit_tags": ["retrieval_fallback"],
            }
            for c in candidates[:top_k]
        ]

    def _run_planner(
        self,
        context: UserQueryContext,
        candidates: list[CandidateScore],
        profile_features: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "context": context.to_dict(),
            "user_profile_features": profile_features,
            "candidate_snapshot": [c.to_compact_dict() for c in candidates[:20]],
        }
        try:
            result = self.llm.chat_json(
                system_prompt=prompts.planner_system_prompt(),
                user_prompt=prompts.planner_user_prompt(payload),
                temperature=0.1,
            )
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return {}

    def run(
        self,
        context: UserQueryContext,
        candidates: list[CandidateScore],
        top_k: int,
        profile_features: dict[str, Any] | None = None,
    ) -> SingleAgentOutput:
        profile_features = profile_features or {}
        allowed_ids = {c.business.business_id for c in candidates}
        planning: dict[str, Any] = {}
        if self.cot_mode == "two_pass":
            planning = self._run_planner(context, candidates, profile_features)

        payload = {
            "context": context.to_dict(),
            "user_profile_features": profile_features,
            "planning": planning,
            "candidates": [c.to_compact_dict() for c in candidates],
            "top_k": top_k,
        }
        try:
            result = self.llm.chat_json(
                system_prompt=prompts.recommender_system_prompt(self.cot_mode),
                user_prompt=prompts.recommender_user_prompt(payload),
                temperature=0.1,
            )
        except Exception:
            result = {}

        recommendations = self._sanitize_recommendations(
            raw_rows=result.get("recommendations", []),
            allowed_ids=allowed_ids,
            top_k=top_k,
        )
        if not recommendations:
            recommendations = self._fallback_recommendations(candidates, top_k)
        else:
            ranked_ids = _dedup_keep_order([row["business_id"] for row in recommendations])
            normalized = []
            seen = set()
            for bid in ranked_ids:
                if bid in seen:
                    continue
                seen.add(bid)
                row = next((x for x in recommendations if x["business_id"] == bid), None)
                if row is not None:
                    normalized.append(row)
            recommendations = normalized[:top_k]

        final_summary = result.get("final_summary", "")
        if not isinstance(final_summary, str) or not final_summary.strip():
            final_summary = (
                "Single-agent reranking over retrieved candidates using query context and profile priors."
            )

        return SingleAgentOutput(
            recommendations=recommendations,
            final_summary=final_summary,
            planning=planning,
        )

