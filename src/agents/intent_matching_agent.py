from __future__ import annotations

from typing import Any

from src.agents.base import AgentSpec, VotingOutput
from src.agents.utils import clip01, semantic_norm, tokenize
from src.schemas import CandidateScore, UserQueryContext


class IntentMatchingAgent:
    @property
    def spec(self) -> AgentSpec:
        return AgentSpec(
            agent_id="A3",
            display_name="Intent Matching Expert",
            description="Evaluate short-term query/session intent alignment.",
            default_weight=0.20,
        )

    def score_candidates(
        self,
        context: UserQueryContext,
        candidates: list[CandidateScore],
        profile_features: dict[str, Any],
    ) -> VotingOutput:
        query_tokens = tokenize(context.query_text)
        scores: dict[str, float] = {}
        confidences: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}

        for c in candidates:
            bid = c.business.business_id
            cand_tokens = tokenize(c.business.name + " " + " ".join(c.business.categories))
            if not query_tokens:
                overlap = 0.0
            else:
                overlap = len(query_tokens & cand_tokens) / len(query_tokens)
            score = clip01(0.65 * semantic_norm(c.text_similarity) + 0.35 * overlap)
            conf = 0.85 if len(query_tokens) >= 3 else 0.65
            tags = ["intent_match"]
            if overlap > 0.25:
                tags.append("keyword_overlap")
            scores[bid] = score
            confidences[bid] = conf
            evidence[bid] = tags

        return VotingOutput(
            agent_id=self.spec.agent_id,
            scores=scores,
            confidences=confidences,
            evidence_tags=evidence,
        )
