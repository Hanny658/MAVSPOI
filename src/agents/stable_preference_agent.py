from __future__ import annotations

from typing import Any

from src.agents.base import AgentSpec
from src.agents.llm_voting_base import LLMVotingAgentBase
from src.agents.utils import clip01, parse_price_level
from src.openai_client import OpenAIService
from src.schemas import CandidateScore, UserQueryContext


class StablePreferenceAgent(LLMVotingAgentBase):
    def __init__(self, llm: OpenAIService, voting_config: dict[str, Any]) -> None:
        super().__init__(llm=llm, voting_config=voting_config)

    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A4",
            display_name="Stable Preference Expert",
            description="Evaluate long-term preference and habit alignment.",
            default_weight=0.17,
        )

    def _expert_focus(self) -> str:
        return "Long-term preference consistency over categories, price range, and habitual patterns."

    def _heuristic_fallback(
        self,
        context: UserQueryContext,
        candidate: CandidateScore,
        profile_features: dict[str, Any],
        extra: dict[str, Any],
    ) -> tuple[float, float, list[str]]:
        top_categories = [
            str(x).strip().lower()
            for x in profile_features.get("top_categories", [])
            if str(x).strip()
        ]
        dominant_price = str(profile_features.get("dominant_price_level", "")).strip()
        support_level = str(profile_features.get("support_level", "unknown")).strip().lower()
        support_conf = {"warm": 0.88, "few_shot": 0.7, "zero_shot": 0.45}.get(
            support_level, 0.5
        )
        cand_categories = [cat.strip().lower() for cat in candidate.business.categories]
        if top_categories:
            cat_match = len(set(top_categories) & set(cand_categories)) / max(
                1, len(set(top_categories))
            )
        else:
            cat_match = 0.5

        price_level = parse_price_level(candidate.business.attributes)
        if dominant_price and price_level:
            price_match = 1.0 if dominant_price == price_level else 0.35
        elif dominant_price or price_level:
            price_match = 0.55
        else:
            price_match = 0.5

        score = clip01(0.75 * cat_match + 0.25 * price_match)
        tags = ["stable_pref"]
        if support_level in {"zero_shot", "unknown"}:
            tags.append("weak_profile")
        return (score, support_conf, tags)
