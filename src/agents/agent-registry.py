"""Compatibility shim.

Python module imports should use `src.agents.agent_registry`.
This file exists to match the requested filename `agent-registry.py`.
"""

from src.agents.agent_registry import *  # noqa: F401,F403
