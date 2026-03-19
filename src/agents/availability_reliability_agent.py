from __future__ import annotations

import math
from typing import Any

from src.agents.base import AgentSpec
from src.agents.llm_voting_base import LLMVotingAgentBase
from src.agents.utils import clip01
from src.openai_client import OpenAIService
from src.schemas import CandidateScore, UserQueryContext


class AvailabilityReliabilityAgent(LLMVotingAgentBase):
    def __init__(self, llm: OpenAIService, voting_config: dict[str, Any]) -> None:
        super().__init__(llm=llm, voting_config=voting_config)

    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A6",
            display_name="Availability-Reliability Expert",
            description="Evaluate candidate validity, freshness proxy, and cold-start risk.",
            default_weight=0.15,
        )

    def _expert_focus(self) -> str:
        return "Operational reliability, metadata completeness, and cold-start risk control."

    def _build_extra(self, candidates: list[CandidateScore]) -> dict[str, Any]:
        max_review = max((c.business.review_count for c in candidates), default=1)
        return {"max_review": max(1, max_review)}

    def _heuristic_fallback(
        self,
        context: UserQueryContext,
        candidate: CandidateScore,
        profile_features: dict[str, Any],
        extra: dict[str, Any],
    ) -> tuple[float, float, list[str]]:
        max_review = int(extra.get("max_review", 1))
        stars_norm = clip01(candidate.business.stars / 5.0)
        review_norm = clip01(
            math.log1p(candidate.business.review_count) / math.log1p(max_review)
        )
        open_norm = 1.0 if int(candidate.business.is_open) == 1 else 0.2
        meta_ok = 1.0 if candidate.business.categories else 0.4
        score = 0.35 * stars_norm + 0.35 * review_norm + 0.2 * open_norm + 0.1 * meta_ok
        tags = ["reliability"]
        conf = 0.85
        if candidate.business.review_count < 5:
            score = clip01(score * 0.9)
            tags.append("cold_start_risk")
            conf = 0.65
        return (score, conf, tags)
