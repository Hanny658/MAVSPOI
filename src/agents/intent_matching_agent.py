from __future__ import annotations

from typing import Any

from src.agents.base import AgentSpec
from src.agents.llm_voting_base import LLMVotingAgentBase
from src.agents.utils import clip01, semantic_norm, tokenize
from src.openai_client import OpenAIService
from src.schemas import CandidateScore, UserQueryContext


class IntentMatchingAgent(LLMVotingAgentBase):
    def __init__(self, llm: OpenAIService, voting_config: dict[str, Any]) -> None:
        super().__init__(llm=llm, voting_config=voting_config)

    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A3",
            display_name="Intent Matching Expert",
            description="Evaluate short-term query/session intent alignment.",
            default_weight=0.20,
        )

    def _expert_focus(self) -> str:
        return "Short-term intent matching between user query/session and candidate semantics."

    def _heuristic_fallback(
        self,
        context: UserQueryContext,
        candidate: CandidateScore,
        profile_features: dict[str, Any],
        extra: dict[str, Any],
    ) -> tuple[float, float, list[str]]:
        query_tokens = tokenize(context.query_text)
        cand_tokens = tokenize(candidate.business.name + " " + " ".join(candidate.business.categories))
        if not query_tokens:
            overlap = 0.0
        else:
            overlap = len(query_tokens & cand_tokens) / len(query_tokens)
        score = clip01(0.65 * semantic_norm(candidate.text_similarity) + 0.35 * overlap)
        conf = 0.85 if len(query_tokens) >= 3 else 0.65
        tags = ["intent_match"]
        if overlap > 0.25:
            tags.append("keyword_overlap")
        return (score, conf, tags)
