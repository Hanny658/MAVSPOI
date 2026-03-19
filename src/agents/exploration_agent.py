from __future__ import annotations

from collections import Counter
from typing import Any

from src.agents.base import AgentSpec, VotingOutput
from src.agents.utils import clip01, semantic_norm
from src.schemas import CandidateScore, UserQueryContext


class ExplorationAgent:
    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A5",
            display_name="Exploration Expert",
            description="Evaluate novelty, diversity contribution, and repetition control.",
            default_weight=0.09,
        )

    def score_candidates(
        self,
        context: UserQueryContext,
        candidates: list[CandidateScore],
        profile_features: dict[str, Any],
    ) -> VotingOutput:
        cat_counter: Counter[str] = Counter()
        for c in candidates:
            for cat in c.business.categories:
                cat_counter[cat.strip().lower()] += 1
        max_cat = max(cat_counter.values(), default=1)

        scores: dict[str, float] = {}
        confidences: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        for c in candidates:
            bid = c.business.business_id
            categories = [x.strip().lower() for x in c.business.categories if x.strip()]
            if categories:
                rarity_components = [1.0 - (cat_counter.get(cat, 0) / max_cat) for cat in categories]
                rarity = sum(rarity_components) / len(rarity_components)
            else:
                rarity = 0.5
            score = clip01(0.7 * rarity + 0.3 * semantic_norm(c.text_similarity))
            tags = ["exploration"]
            if rarity > 0.5:
                tags.append("novel_category")
            scores[bid] = score
            confidences[bid] = 0.55
            evidence[bid] = tags

        return VotingOutput(
            agent_id=self.spec.agent_id,
            scores=scores,
            confidences=confidences,
            evidence_tags=evidence,
        )
