from __future__ import annotations

from typing import Any

from src.agents.base import AgentSpec, VotingOutput
from src.agents.utils import clip01, parse_bool_text, purpose_label, semantic_norm, tokenize
from src.schemas import CandidateScore, UserQueryContext


class PurposeModalityAgent:
    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A7",
            display_name="Purpose-Modality Expert",
            description="Evaluate purpose fit (social/study/work) and modality suitability.",
            default_weight=0.09,
        )

    def score_candidates(
        self,
        context: UserQueryContext,
        candidates: list[CandidateScore],
        profile_features: dict[str, Any],
    ) -> VotingOutput:
        query_tokens = tokenize(context.query_text)
        purpose = purpose_label(query_tokens)
        study_cats = {"coffee", "cafes", "cafe", "coworking", "library", "tea"}
        social_cats = {"bars", "nightlife", "lounges", "restaurants", "karaoke"}
        family_cats = {"family", "parks", "playgrounds", "zoos", "museums"}
        delivery_cats = {"food", "restaurants", "pizza", "sandwiches", "chinese", "mexican"}

        scores: dict[str, float] = {}
        confidences: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        for c in candidates:
            bid = c.business.business_id
            cand_tokens = tokenize(" ".join(c.business.categories))
            attrs = c.business.attributes or {}

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

            score = clip01(0.7 * match + 0.3 * semantic_norm(c.text_similarity))
            conf = 0.8 if purpose != "generic" else 0.45
            scores[bid] = score
            confidences[bid] = conf
            evidence[bid] = ["purpose_modality", purpose]

        return VotingOutput(
            agent_id=self.spec.agent_id,
            scores=scores,
            confidences=confidences,
            evidence_tags=evidence,
        )
