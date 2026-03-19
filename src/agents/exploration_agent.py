from __future__ import annotations

from collections import Counter
from typing import Any

from src.agents.base import AgentSpec
from src.agents.llm_voting_base import LLMVotingAgentBase
from src.agents.utils import clip01, semantic_norm
from src.openai_client import OpenAIService
from src.schemas import CandidateScore, UserQueryContext


class ExplorationAgent(LLMVotingAgentBase):
    def __init__(self, llm: OpenAIService, voting_config: dict[str, Any]) -> None:
        super().__init__(llm=llm, voting_config=voting_config)

    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A5",
            display_name="Exploration Expert",
            description="Evaluate novelty, diversity contribution, and repetition control.",
            default_weight=0.09,
        )

    def _expert_focus(self) -> str:
        return "Novelty and diversity gains while avoiding excessive relevance loss."

    def _build_extra(self, candidates: list[CandidateScore]) -> dict[str, Any]:
        cat_counter: Counter[str] = Counter()
        for c in candidates:
            for cat in c.business.categories:
                cat_counter[cat.strip().lower()] += 1
        return {"cat_counter": cat_counter, "max_cat": max(cat_counter.values(), default=1)}

    def _heuristic_fallback(
        self,
        context: UserQueryContext,
        candidate: CandidateScore,
        profile_features: dict[str, Any],
        extra: dict[str, Any],
    ) -> tuple[float, float, list[str]]:
        cat_counter = extra.get("cat_counter", Counter())
        max_cat = max(1, int(extra.get("max_cat", 1)))
        categories = [x.strip().lower() for x in candidate.business.categories if x.strip()]
        if categories:
            rarity_components = [1.0 - (cat_counter.get(cat, 0) / max_cat) for cat in categories]
            rarity = sum(rarity_components) / len(rarity_components)
        else:
            rarity = 0.5
        score = clip01(0.7 * rarity + 0.3 * semantic_norm(candidate.text_similarity))
        tags = ["exploration"]
        if rarity > 0.5:
            tags.append("novel_category")
        return (score, 0.55, tags)
