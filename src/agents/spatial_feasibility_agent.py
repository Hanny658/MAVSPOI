from __future__ import annotations

from typing import Any

from src.agents.base import AgentSpec, VotingOutput
from src.agents.llm_voting_base import LLMVotingAgentBase
from src.agents.utils import clip01, exp_distance_score, semantic_norm
from src.openai_client import OpenAIService
from src.schemas import CandidateScore, UserQueryContext


class SpatialFeasibilityAgent(LLMVotingAgentBase):
    def __init__(self, llm: OpenAIService, voting_config: dict[str, Any]) -> None:
        super().__init__(llm=llm, voting_config=voting_config)

    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A1",
            display_name="Spatial Feasibility Expert",
            description="Evaluate spatial reachability and detour cost.",
            default_weight=0.16,
        )

    def _expert_focus(self) -> str:
        return "Spatial reachability, detour cost, and distance tolerance."

    def _heuristic_fallback(
        self,
        context: UserQueryContext,
        candidate: CandidateScore,
        profile_features: dict[str, Any],
        extra: dict[str, Any],
    ) -> tuple[float, float, list[str]]:
        radius = float(profile_features.get("radius_km_p90", 10.0) or 10.0)
        radius = min(30.0, max(2.0, radius))
        if candidate.distance_km is None:
            return (0.55, 0.35, ["no_geo"])
        dist_score = exp_distance_score(candidate.distance_km, radius)
        score = clip01(0.75 * dist_score + 0.25 * semantic_norm(candidate.text_similarity))
        tags = ["nearby"] if candidate.distance_km <= 2.0 else ["spatial_fit"]
        return (score, 0.9, tags)
