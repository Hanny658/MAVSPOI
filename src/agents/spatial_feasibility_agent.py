from __future__ import annotations

from typing import Any

from src.agents.base import AgentSpec, VotingOutput
from src.agents.utils import clip01, exp_distance_score, semantic_norm
from src.schemas import CandidateScore, UserQueryContext


class SpatialFeasibilityAgent:
    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A1",
            display_name="Spatial Feasibility Expert",
            description="Evaluate spatial reachability and detour cost.",
            default_weight=0.16,
        )

    def score_candidates(
        self,
        context: UserQueryContext,
        candidates: list[CandidateScore],
        profile_features: dict[str, Any],
    ) -> VotingOutput:
        scores: dict[str, float] = {}
        confidences: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}

        radius = float(profile_features.get("radius_km_p90", 10.0) or 10.0)
        radius = min(30.0, max(2.0, radius))
        for c in candidates:
            bid = c.business.business_id
            if c.distance_km is None:
                score = 0.55
                conf = 0.35
                tags = ["no_geo"]
            else:
                dist_score = exp_distance_score(c.distance_km, radius)
                score = clip01(0.75 * dist_score + 0.25 * semantic_norm(c.text_similarity))
                conf = 0.9
                tags = ["nearby"] if c.distance_km <= 2.0 else ["spatial_fit"]
            scores[bid] = score
            confidences[bid] = conf
            evidence[bid] = tags

        return VotingOutput(
            agent_id=self.spec.agent_id,
            scores=scores,
            confidences=confidences,
            evidence_tags=evidence,
        )
