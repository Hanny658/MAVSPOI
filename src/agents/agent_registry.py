from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from src.agents.availability_reliability_agent import AvailabilityReliabilityAgent
from src.agents.base import AgentSpec, VotingAgent
from src.agents.exploration_agent import ExplorationAgent
from src.agents.intent_matching_agent import IntentMatchingAgent
from src.agents.purpose_modality_agent import PurposeModalityAgent
from src.agents.spatial_feasibility_agent import SpatialFeasibilityAgent
from src.agents.stable_preference_agent import StablePreferenceAgent
from src.agents.temporal_feasibility_agent import TemporalFeasibilityAgent


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, VotingAgent] = {}

    def register(self, agent: VotingAgent) -> None:
        self._agents[agent.spec.agent_id] = agent

    def register_many(self, agents: Iterable[VotingAgent]) -> None:
        for agent in agents:
            self.register(agent)

    def get(self, agent_id: str) -> VotingAgent | None:
        return self._agents.get(agent_id)

    def list_specs(self) -> list[AgentSpec]:
        return [agent.spec for agent in self._agents.values()]

    def list_specs_as_dict(self) -> list[dict]:
        return [asdict(spec) for spec in self.list_specs()]

    def enabled(self, enabled_ids: list[str] | None = None) -> list[VotingAgent]:
        if not enabled_ids:
            return list(self._agents.values())
        out: list[VotingAgent] = []
        for aid in enabled_ids:
            agent = self._agents.get(aid)
            if agent is not None:
                out.append(agent)
        return out


def build_default_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register_many(
        [
            SpatialFeasibilityAgent(),
            TemporalFeasibilityAgent(),
            IntentMatchingAgent(),
            StablePreferenceAgent(),
            ExplorationAgent(),
            AvailabilityReliabilityAgent(),
            PurposeModalityAgent(),
        ]
    )
    return registry
