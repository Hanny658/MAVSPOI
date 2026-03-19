from __future__ import annotations

import json
from typing import Any

from src.agents.agent_registry import AgentRegistry
from src.agents.base import RouterActivation, RouterDecision
from src.agents.utils import clip01, purpose_label, safe_float, tokenize
from src.openai_client import OpenAIService
from src.schemas import CandidateScore, UserQueryContext


class RouterAgent:
    def __init__(
        self,
        llm: OpenAIService,
        registry: AgentRegistry,
        config: dict[str, Any],
        base_weights: dict[str, float],
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.config = config
        self.base_weights = base_weights

    def _heuristic_scores(
        self,
        context: UserQueryContext,
        profile_features: dict[str, Any],
    ) -> dict[str, tuple[float, str]]:
        query_tokens = tokenize(context.query_text)
        support_level = str(profile_features.get("support_level", "unknown")).strip().lower()
        has_location = context.location is not None
        has_time_signal = bool(
            query_tokens
            & {"now", "today", "tonight", "lunch", "dinner", "breakfast", "urgent", "asap"}
        ) or bool(context.local_time.strip())
        purpose = purpose_label(query_tokens)

        return {
            "A1": (
                0.85 if has_location else 0.35,
                "Location-aware query." if has_location else "No precise location.",
            ),
            "A2": (
                0.8 if has_time_signal else 0.45,
                "Temporal cues detected." if has_time_signal else "Weak temporal cues.",
            ),
            "A3": (0.8, "Intent matching is core for all queries."),
            "A4": (
                0.78 if support_level == "warm" else (0.62 if support_level == "few_shot" else 0.45),
                f"Profile support level: {support_level}.",
            ),
            "A5": (0.52, "Exploration as secondary signal."),
            "A6": (0.72, "Reliability guardrail enabled."),
            "A7": (
                0.75 if purpose != "generic" else 0.48,
                f"Purpose inferred: {purpose}.",
            ),
        }

    def _llm_scores(self, payload: dict[str, Any]) -> dict[str, float]:
        if not bool(self.config.get("use_llm", False)):
            return {}
        system_prompt = (
            "You are RouterAgent in a Mixture-of-Agents POI recommender. "
            "Return JSON only: {\"agent_scores\": {\"A1\":0,...,\"A7\":0}} with scores in [0,1]."
        )
        try:
            result = self.llm.chat_json(
                system_prompt=system_prompt,
                user_prompt="Input JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2),
                temperature=safe_float(self.config.get("llm_temperature", 0.1), 0.1),
            )
        except Exception:
            return {}
        raw = result.get("agent_scores", {}) if isinstance(result, dict) else {}
        out: dict[str, float] = {}
        if isinstance(raw, dict):
            for aid, score in raw.items():
                out[str(aid)] = clip01(safe_float(score, 0.0))
        return out

    def run(
        self,
        context: UserQueryContext,
        profile_features: dict[str, Any],
        candidates: list[CandidateScore],
    ) -> RouterDecision:
        enabled_ids = list(self.config.get("enabled_agents", [])) or [
            spec.agent_id for spec in self.registry.list_specs()
        ]
        heuristic = self._heuristic_scores(context, profile_features)
        payload = {
            "context": context.to_dict(),
            "profile_features": profile_features,
            "candidate_snapshot": [c.to_compact_dict() for c in candidates[:15]],
            "agent_specs": self.registry.list_specs_as_dict(),
            "heuristic_scores": {k: v[0] for k, v in heuristic.items()},
        }
        llm_scores = self._llm_scores(payload)

        threshold = safe_float(self.config.get("activation_threshold", 0.45), 0.45)
        min_agents = int(safe_float(self.config.get("min_agents", 3), 3))
        max_agents = int(safe_float(self.config.get("max_agents", 5), 5))
        fallback_agents = list(self.config.get("fallback_agents", ["A1", "A3", "A4", "A6"]))

        ranked: list[tuple[str, float, str]] = []
        for aid in enabled_ids:
            h_score, reason = heuristic.get(aid, (0.5, "No heuristic signal."))
            if aid in llm_scores:
                score = 0.5 * h_score + 0.5 * llm_scores[aid]
                reason = reason + " LLM-merged."
            else:
                score = h_score
            ranked.append((aid, clip01(score), reason))
        ranked.sort(key=lambda x: x[1], reverse=True)

        selected = [row for row in ranked if row[1] >= threshold]
        if len(selected) < min_agents:
            selected = ranked[:min_agents]
        selected = selected[:max_agents]
        if not selected:
            selected = [(aid, 0.5, "Fallback activation.") for aid in fallback_agents[:min_agents]]

        raw_weights: list[float] = []
        for aid, score, _ in selected:
            base_w = safe_float(self.base_weights.get(aid, 1.0), 1.0)
            raw_weights.append(max(0.01, base_w * max(score, 0.05)))
        total_weight = sum(raw_weights) or 1.0

        activations: list[RouterActivation] = []
        for (aid, score, reason), raw_w in zip(selected, raw_weights):
            activations.append(
                RouterActivation(
                    agent_id=aid,
                    weight=raw_w / total_weight,
                    confidence=score,
                    reason=reason,
                )
            )

        support_level = str(profile_features.get("support_level", "unknown")).strip().lower()
        risk_flags: list[str] = []
        if support_level in {"zero_shot", "unknown"}:
            risk_flags.append("low_profile_support")
        if context.location is None:
            risk_flags.append("missing_location")
        if len(tokenize(context.query_text)) <= 2:
            risk_flags.append("ambiguous_query")

        radius_p90 = safe_float(profile_features.get("radius_km_p90", 0.0), 0.0)
        max_distance = safe_float(self.config.get("default_max_distance_km", 10.0), 10.0)
        if radius_p90 > 0:
            max_distance = max(2.0, min(30.0, radius_p90 * 1.5))

        query_tokens = tokenize(context.query_text)
        open_now = bool(query_tokens & {"now", "today", "tonight", "immediately", "asap"})
        constraints = {
            "city": context.city,
            "state": context.state,
            "open_now": open_now,
            "max_distance_km": round(max_distance, 3),
        }
        return RouterDecision(
            activated_agents=activations,
            global_constraints=constraints,
            risk_flags=risk_flags,
        )
