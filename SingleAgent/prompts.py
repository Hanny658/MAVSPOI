from __future__ import annotations

import json
from typing import Any


def planner_system_prompt() -> str:
    return (
        "You are a planning module inside a single-agent POI recommender.\n"
        "Goal: produce concise ranking guidance from context/profile/candidates.\n"
        "Rules:\n"
        "- Use only the provided evidence.\n"
        "- Keep guidance short and operational.\n"
        "- Output valid JSON only.\n"
        "JSON schema:\n"
        "{\n"
        '  "query_intent": "string",\n'
        '  "hard_constraints": ["string"],\n'
        '  "soft_preferences": ["string"],\n'
        '  "ranking_strategy": ["string"]\n'
        "}"
    )


def planner_user_prompt(payload: dict[str, Any]) -> str:
    return "Input JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def recommender_system_prompt(cot_mode: str) -> str:
    cot_rule = ""
    if cot_mode == "embedded":
        cot_rule = (
            "- Think through trade-offs internally before ranking, but do not output "
            "hidden reasoning text.\n"
        )
    return (
        "You are a single-agent POI recommender.\n"
        "Task: rerank retrieved candidates and output final Top-N results.\n"
        "Rules:\n"
        "- business_id must come from provided candidates only.\n"
        "- Integrate user_profile_features and query context.\n"
        "- Prioritize practical suitability for current time/location/request.\n"
        f"{cot_rule}"
        "- Output valid JSON only.\n"
        "JSON schema:\n"
        "{\n"
        '  "recommendations": [\n'
        "    {\n"
        '      "business_id": "string",\n'
        '      "score": 0,\n'
        '      "reason": "string",\n'
        '      "fit_tags": ["string"]\n'
        "    }\n"
        "  ],\n"
        '  "final_summary": "string"\n'
        "}"
    )


def recommender_user_prompt(payload: dict[str, Any]) -> str:
    return "Input JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

