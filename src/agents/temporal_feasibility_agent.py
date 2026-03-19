from __future__ import annotations

from typing import Any

from src.agents.base import AgentSpec
from src.agents.llm_voting_base import LLMVotingAgentBase
from src.agents.utils import clip01, parse_hour, semantic_norm, tokenize
from src.openai_client import OpenAIService
from src.schemas import CandidateScore, UserQueryContext


class TemporalFeasibilityAgent(LLMVotingAgentBase):
    def __init__(self, llm: OpenAIService, voting_config: dict[str, Any]) -> None:
        super().__init__(llm=llm, voting_config=voting_config)

    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A2",
            display_name="Temporal Feasibility Expert",
            description="Evaluate time-window fit, urgency, and period suitability.",
            default_weight=0.14,
        )

    def _expert_focus(self) -> str:
        return "Time fit, urgency compatibility, and likely availability in current time window."

    def _heuristic_fallback(
        self,
        context: UserQueryContext,
        candidate: CandidateScore,
        profile_features: dict[str, Any],
        extra: dict[str, Any],
    ) -> tuple[float, float, list[str]]:
        hour = parse_hour(context.local_time)
        active_hours = profile_features.get("active_hours_top3", [])
        active_set = {
            int(x)
            for x in active_hours
            if isinstance(x, (int, float, str)) and str(x).isdigit()
        }
        urgent = bool(
            tokenize(context.query_text)
            & {"now", "urgent", "asap", "immediately", "quick", "right"}
        )
        time_fit = 0.6
        if hour is not None and active_set:
            time_fit = 1.0 if hour in active_set else 0.45
        elif hour is not None:
            time_fit = 0.7
        open_score = 1.0 if int(candidate.business.is_open) == 1 else 0.25
        urgency_bonus = 0.1 if urgent and int(candidate.business.is_open) == 1 else 0.0
        score = clip01(
            0.5 * time_fit
            + 0.25 * open_score
            + 0.25 * semantic_norm(candidate.text_similarity)
            + urgency_bonus
        )
        tags = ["time_fit"]
        if urgent:
            tags.append("urgent")
        return (score, 0.78 if hour is not None else 0.55, tags)
