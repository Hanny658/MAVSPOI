from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.schemas import CandidateScore, UserQueryContext


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    display_name: str
    description: str
    default_weight: float


@dataclass
class RouterActivation:
    agent_id: str
    weight: float
    confidence: float
    reason: str


@dataclass
class RouterDecision:
    activated_agents: list[RouterActivation]
    global_constraints: dict[str, Any]
    risk_flags: list[str]


@dataclass
class VotingOutput:
    agent_id: str
    scores: dict[str, float]
    confidences: dict[str, float]
    evidence_tags: dict[str, list[str]]
    status: str = "ok"


@dataclass
class AggregatedRecommendation:
    business_id: str
    final_score: float
    reason: str
    fit_tags: list[str]
    contribution: dict[str, float]


@dataclass
class AggregationOutput:
    recommendations: list[AggregatedRecommendation]
    summary: str


class VotingAgent(Protocol):
    @property
    def spec(self) -> AgentSpec: ...

    def score_candidates(
        self,
        context: UserQueryContext,
        candidates: list[CandidateScore],
        profile_features: dict[str, Any],
    ) -> VotingOutput: ...
