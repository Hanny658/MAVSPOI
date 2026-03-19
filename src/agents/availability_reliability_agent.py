from __future__ import annotations

import math
from typing import Any

from src.agents.base import AgentSpec, VotingOutput
from src.agents.utils import clip01
from src.schemas import CandidateScore, UserQueryContext


class AvailabilityReliabilityAgent:
    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A6",
            display_name="Availability-Reliability Expert",
            description="Evaluate candidate validity, freshness proxy, and cold-start risk.",
            default_weight=0.15,
        )

    def score_candidates(
        self,
        context: UserQueryContext,
        candidates: list[CandidateScore],
        profile_features: dict[str, Any],
    ) -> VotingOutput:
        max_review = max((c.business.review_count for c in candidates), default=1)
        max_review = max(1, max_review)

        scores: dict[str, float] = {}
        confidences: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        for c in candidates:
            bid = c.business.business_id
            stars_norm = clip01(c.business.stars / 5.0)
            review_norm = clip01(
                math.log1p(c.business.review_count) / math.log1p(max_review)
            )
            open_norm = 1.0 if int(c.business.is_open) == 1 else 0.2
            meta_ok = 1.0 if c.business.categories else 0.4
            score = 0.35 * stars_norm + 0.35 * review_norm + 0.2 * open_norm + 0.1 * meta_ok
            tags = ["reliability"]
            conf = 0.85
            if c.business.review_count < 5:
                score = clip01(score * 0.9)
                tags.append("cold_start_risk")
                conf = 0.65
            scores[bid] = score
            confidences[bid] = conf
            evidence[bid] = tags

        return VotingOutput(
            agent_id=self.spec.agent_id,
            scores=scores,
            confidences=confidences,
            evidence_tags=evidence,
        )
