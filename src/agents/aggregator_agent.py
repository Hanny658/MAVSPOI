from __future__ import annotations

from collections import Counter
from typing import Any

from src.agents.base import (
    AggregatedRecommendation,
    AggregationOutput,
    RouterDecision,
    VotingOutput,
)
from src.agents.utils import clip01, dedup_keep_order, safe_float
from src.schemas import CandidateScore


class AggregatorAgent:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.retrieval_weight = clip01(safe_float(config.get("retrieval_weight", 0.25), 0.25))
        self.diversity_penalty = clip01(safe_float(config.get("diversity_penalty", 0.03), 0.04))
        self.neutral_score = clip01(safe_float(config.get("neutral_score", 0.5), 0.5))

    def _retrieval_norm_map(self, candidates: list[CandidateScore]) -> dict[str, float]:
        if not candidates:
            return {}
        vals = [float(c.score) for c in candidates]
        lo, hi = min(vals), max(vals)
        out: dict[str, float] = {}
        for c in candidates:
            bid = c.business.business_id
            if hi - lo <= 1e-9:
                out[bid] = 0.5
            else:
                out[bid] = clip01((float(c.score) - lo) / (hi - lo))
        return out

    def _primary_category(self, candidate: CandidateScore) -> str:
        cats = candidate.business.categories
        if not cats:
            return "unknown"
        return cats[0].strip().lower() or "unknown"

    def _diversity_rerank(
        self,
        rows: list[dict[str, Any]],
        by_id: dict[str, CandidateScore],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if self.diversity_penalty <= 1e-9:
            return rows[:top_k]
        remaining = list(rows)
        selected: list[dict[str, Any]] = []
        category_count: Counter[str] = Counter()
        while remaining and len(selected) < top_k:
            best_idx = 0
            best_adjusted = -1e9
            for idx, row in enumerate(remaining):
                candidate = by_id[row["business_id"]]
                category = self._primary_category(candidate)
                adjusted = float(row["final_score"]) - self.diversity_penalty * category_count[category]
                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_idx = idx
            chosen = remaining.pop(best_idx)
            candidate = by_id[chosen["business_id"]]
            category_count[self._primary_category(candidate)] += 1
            chosen["final_score"] = clip01(best_adjusted)
            selected.append(chosen)
        return selected

    def run(
        self,
        router_decision: RouterDecision,
        votes: dict[str, VotingOutput],
        candidates: list[CandidateScore],
        top_k: int,
        constraint_penalties: dict[str, float] | None = None,
        constraint_tags: dict[str, list[str]] | None = None,
    ) -> AggregationOutput:
        if not candidates:
            return AggregationOutput(recommendations=[], summary="No candidates available.")

        constraint_penalties = constraint_penalties or {}
        constraint_tags = constraint_tags or {}
        by_id = {c.business.business_id: c for c in candidates}
        retrieval_norm = self._retrieval_norm_map(candidates)
        risk_flags = set(router_decision.risk_flags)
        risk_boost = 0.0
        if "low_profile_support" in risk_flags:
            risk_boost += 0.08
        if "low_query_specificity" in risk_flags:
            risk_boost += 0.06
        effective_retrieval_weight = clip01(self.retrieval_weight + risk_boost)
        rows: list[dict[str, Any]] = []

        for candidate in candidates:
            bid = candidate.business.business_id
            contributions: dict[str, float] = {}
            total = 0.0

            retrieval_part = effective_retrieval_weight * retrieval_norm.get(bid, 0.5)
            contributions["retrieval"] = round(retrieval_part, 6)
            total += retrieval_part

            fit_tags: list[str] = []
            for activation in router_decision.activated_agents:
                vote = votes.get(activation.agent_id)
                if vote is None:
                    continue
                score = vote.scores.get(bid, self.neutral_score)
                conf = vote.confidences.get(bid, 0.5)
                routing_reliability = 0.5 + 0.5 * clip01(float(activation.confidence))
                part = float(activation.weight) * routing_reliability * float(conf) * float(score)
                contributions[activation.agent_id] = round(part, 6)
                total += part
                fit_tags.extend(vote.evidence_tags.get(bid, []))

            penalty = clip01(safe_float(constraint_penalties.get(bid, 0.0), 0.0))
            if penalty > 0:
                contributions["constraint_penalty"] = -round(penalty, 6)
                fit_tags.extend(constraint_tags.get(bid, []))
                total -= penalty

            total = clip01(total)
            top_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
            reason = "Top factors: " + ", ".join(f"{k}={v:.3f}" for k, v in top_factors)
            if penalty > 0:
                reason += f". Constraint penalty={penalty:.3f}"
            rows.append(
                {
                    "business_id": bid,
                    "final_score": total,
                    "reason": reason,
                    "fit_tags": dedup_keep_order(fit_tags)[:6],
                    "contribution": contributions,
                }
            )

        rows.sort(key=lambda x: x["final_score"], reverse=True)
        ranked_rows = self._diversity_rerank(rows, by_id=by_id, top_k=top_k)
        recommendations = [
            AggregatedRecommendation(
                business_id=row["business_id"],
                final_score=row["final_score"],
                reason=row["reason"],
                fit_tags=row["fit_tags"],
                contribution=row["contribution"],
            )
            for row in ranked_rows
        ]
        agent_names = ", ".join(a.agent_id for a in router_decision.activated_agents)
        summary = (
            "Deterministic aggregation over retrieval and activated voting experts. "
            f"Activated agents: {agent_names}."
        )
        return AggregationOutput(recommendations=recommendations, summary=summary)
