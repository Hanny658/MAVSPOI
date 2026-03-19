from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from src.agents.base import AgentSpec, VotingOutput
from src.agents.utils import clip01, dedup_keep_order, safe_float
from src.openai_client import OpenAIService
from src.schemas import CandidateScore, UserQueryContext


class LLMVotingAgentBase(ABC):
    def __init__(self, llm: OpenAIService, voting_config: dict[str, Any]) -> None:
        self.llm = llm
        self.voting_config = voting_config

    @property
    @abstractmethod
    def spec(self) -> AgentSpec:
        raise NotImplementedError

    @abstractmethod
    def _heuristic_fallback(
        self,
        context: UserQueryContext,
        candidate: CandidateScore,
        profile_features: dict[str, Any],
        extra: dict[str, Any],
    ) -> tuple[float, float, list[str]]:
        raise NotImplementedError

    @abstractmethod
    def _expert_focus(self) -> str:
        raise NotImplementedError

    def _candidate_snapshot(
        self,
        candidates: list[CandidateScore],
        limit: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for c in candidates[:limit]:
            row = c.to_compact_dict()
            row["distance_km"] = c.distance_km
            rows.append(row)
        return rows

    def _build_system_prompt(self) -> str:
        return (
            f"You are {self.spec.display_name} in a Mixture-of-Agents voting recommender.\n"
            f"Responsibility: {self._expert_focus()}\n"
            "Think step by step internally, but do not reveal hidden reasoning.\n"
            "Output valid JSON only.\n"
            "Score range: [0,1]. Confidence range: [0,1].\n"
            "You may score fewer candidates if uncertain; missing ones will be handled by fallback.\n"
            "JSON schema:\n"
            "{\n"
            '  "results": [\n'
            "    {\n"
            '      "business_id": "string",\n'
            '      "score": 0,\n'
            '      "confidence": 0,\n'
            '      "evidence_tags": ["string"]\n'
            "    }\n"
            "  ]\n"
            "}"
        )

    def _llm_vote(
        self,
        context: UserQueryContext,
        candidates: list[CandidateScore],
        profile_features: dict[str, Any],
    ) -> dict[str, Any]:
        if not bool(self.voting_config.get("llm_enabled", True)):
            return {}
        payload = {
            "context": context.to_dict(),
            "profile_features": profile_features,
            "agent_id": self.spec.agent_id,
            "expert_focus": self._expert_focus(),
            "candidates": self._candidate_snapshot(
                candidates,
                limit=int(safe_float(self.voting_config.get("llm_candidate_limit", 30), 30)),
            ),
        }
        try:
            return self.llm.chat_json(
                system_prompt=self._build_system_prompt(),
                user_prompt="Input JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2),
                temperature=safe_float(self.voting_config.get("llm_temperature", 0.1), 0.1),
                max_tokens=int(safe_float(self.voting_config.get("llm_max_tokens", 1600), 1600)),
            )
        except Exception:
            return {}

    def score_candidates(
        self,
        context: UserQueryContext,
        candidates: list[CandidateScore],
        profile_features: dict[str, Any],
    ) -> VotingOutput:
        extra = self._build_extra(candidates)
        heuristic_scores: dict[str, float] = {}
        heuristic_conf: dict[str, float] = {}
        heuristic_tags: dict[str, list[str]] = {}
        for c in candidates:
            bid = c.business.business_id
            s, conf, tags = self._heuristic_fallback(context, c, profile_features, extra)
            heuristic_scores[bid] = clip01(s)
            heuristic_conf[bid] = clip01(conf)
            heuristic_tags[bid] = tags

        llm_out = self._llm_vote(context, candidates, profile_features)
        llm_map: dict[str, tuple[float, float, list[str]]] = {}
        raw_results = llm_out.get("results", []) if isinstance(llm_out, dict) else []
        if isinstance(raw_results, list):
            allowed = {c.business.business_id for c in candidates}
            for row in raw_results:
                if not isinstance(row, dict):
                    continue
                bid = str(row.get("business_id", "")).strip()
                if bid not in allowed:
                    continue
                score = clip01(safe_float(row.get("score", 0.5), 0.5))
                conf = clip01(safe_float(row.get("confidence", 0.5), 0.5))
                tags_raw = row.get("evidence_tags", [])
                tags = [str(x) for x in tags_raw] if isinstance(tags_raw, list) else []
                llm_map[bid] = (score, conf, tags[:6])

        h_weight = clip01(safe_float(self.voting_config.get("heuristic_weight", 0.35), 0.35))
        l_weight = clip01(safe_float(self.voting_config.get("llm_weight", 0.65), 0.65))
        if not llm_map:
            h_weight = 1.0
            l_weight = 0.0
        total_w = max(1e-6, h_weight + l_weight)

        final_scores: dict[str, float] = {}
        final_conf: dict[str, float] = {}
        final_tags: dict[str, list[str]] = {}
        for c in candidates:
            bid = c.business.business_id
            hs = heuristic_scores[bid]
            hc = heuristic_conf[bid]
            htags = heuristic_tags[bid]
            ls, lc, ltags = llm_map.get(bid, (hs, hc * 0.8, []))

            score = clip01((h_weight * hs + l_weight * ls) / total_w)
            conf = clip01((h_weight * hc + l_weight * lc) / total_w)
            tags = dedup_keep_order(htags + ltags)[:6]

            final_scores[bid] = score
            final_conf[bid] = conf
            final_tags[bid] = tags

        return VotingOutput(
            agent_id=self.spec.agent_id,
            scores=final_scores,
            confidences=final_conf,
            evidence_tags=final_tags,
            status="ok",
        )

    def _build_extra(self, candidates: list[CandidateScore]) -> dict[str, Any]:
        return {}
