from __future__ import annotations

from typing import Any

from src.agents.base import AgentSpec
from src.agents.llm_voting_base import LLMVotingAgentBase
from src.agents.utils import clip01, parse_bool_text, purpose_label, semantic_norm, tokenize
from src.openai_client import OpenAIService
from src.schemas import CandidateScore, UserQueryContext


class PurposeModalityAgent(LLMVotingAgentBase):
    def __init__(self, llm: OpenAIService, voting_config: dict[str, Any]) -> None:
        super().__init__(llm=llm, voting_config=voting_config)

    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A7",
            display_name="Purpose-Modality Expert",
            description="Evaluate purpose fit (social/study/work) and modality suitability.",
            default_weight=0.09,
        )

    def _expert_focus(self) -> str:
        return "Purpose-level scenario fit (social/study/work/family/delivery) and modality suitability."

    def _heuristic_fallback(
        self,
        context: UserQueryContext,
        candidate: CandidateScore,
        profile_features: dict[str, Any],
        extra: dict[str, Any],
    ) -> tuple[float, float, list[str]]:
        query_tokens = tokenize(context.query_text)
        purpose = purpose_label(query_tokens)
        study_cats = {"coffee", "cafes", "cafe", "coworking", "library", "tea"}
        social_cats = {"bars", "nightlife", "lounges", "restaurants", "karaoke"}
        family_cats = {"family", "parks", "playgrounds", "zoos", "museums"}
        delivery_cats = {"food", "restaurants", "pizza", "sandwiches", "chinese", "mexican"}
        cand_tokens = tokenize(" ".join(candidate.business.categories))
        attrs = candidate.business.attributes or {}

        if purpose == "study_work":
            match = 1.0 if cand_tokens & study_cats else 0.4
            if parse_bool_text(attrs.get("WiFi", "")):
                match = min(1.0, match + 0.15)
        elif purpose == "social":
            match = 1.0 if cand_tokens & social_cats else 0.45
        elif purpose == "family":
            match = 1.0 if cand_tokens & family_cats else 0.45
        elif purpose == "delivery":
            modality = parse_bool_text(attrs.get("RestaurantsDelivery", "")) or parse_bool_text(
                attrs.get("RestaurantsTakeOut", "")
            )
            cat_match = 1.0 if cand_tokens & delivery_cats else 0.45
            match = min(1.0, 0.75 * cat_match + 0.25 * (1.0 if modality else 0.4))
        else:
            match = 0.55

        score = clip01(0.7 * match + 0.3 * semantic_norm(candidate.text_similarity))
        conf = 0.8 if purpose != "generic" else 0.45
        return (score, conf, ["purpose_modality", purpose])
