from __future__ import annotations

from typing import Any

from src.agents.base import AgentSpec, VotingOutput
from src.agents.utils import clip01, parse_price_level
from src.schemas import CandidateScore, UserQueryContext


class StablePreferenceAgent:
    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A4",
            display_name="Stable Preference Expert",
            description="Evaluate long-term preference and habit alignment.",
            default_weight=0.17,
        )

    def score_candidates(
        self,
        context: UserQueryContext,
        candidates: list[CandidateScore],
        profile_features: dict[str, Any],
    ) -> VotingOutput:
        top_categories = [
            str(x).strip().lower()
            for x in profile_features.get("top_categories", [])
            if str(x).strip()
        ]
        dominant_price = str(profile_features.get("dominant_price_level", "")).strip()
        support_level = str(profile_features.get("support_level", "unknown")).strip().lower()

        scores: dict[str, float] = {}
        confidences: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}

        support_conf = {"warm": 0.88, "few_shot": 0.7, "zero_shot": 0.45}.get(
            support_level, 0.5
        )
        for c in candidates:
            bid = c.business.business_id
            cand_categories = [cat.strip().lower() for cat in c.business.categories]
            if top_categories:
                cat_match = len(set(top_categories) & set(cand_categories)) / max(
                    1, len(set(top_categories))
                )
            else:
                cat_match = 0.5

            price_level = parse_price_level(c.business.attributes)
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
            scores[bid] = score
            confidences[bid] = support_conf
            evidence[bid] = tags

        return VotingOutput(
            agent_id=self.spec.agent_id,
            scores=scores,
            confidences=confidences,
            evidence_tags=evidence,
        )
