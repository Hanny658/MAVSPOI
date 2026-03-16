from __future__ import annotations

import json
from typing import Any


def profiler_system_prompt() -> str:
    return (
        "You are ProfilerAgent in a CoMaPOI-styled multi-agent recommendation system.\n"
        "Goal: transform structured user request into two language layers:\n"
        "1) long-term profile (macro preferences), 2) short-term mobility/intent pattern.\n"
        "Rules:\n"
        "- Use only provided input facts.\n"
        "- If user_profile_features are provided, treat them as historical evidence.\n"
        "- Keep outputs concise and operational for downstream ranking.\n"
        "- Do not invent unavailable user attributes.\n"
        "- Output valid JSON only.\n"
        "JSON schema:\n"
        "{\n"
        '  "long_term_profile": {\n'
        '    "summary": "string",\n'
        '    "preferred_categories": ["string"],\n'
        '    "price_preference": "string",\n'
        '    "distance_preference_km": 0,\n'
        '    "avoid_categories": ["string"]\n'
        "  },\n"
        '  "short_term_pattern": {\n'
        '    "intent_summary": "string",\n'
        '    "time_sensitivity": "string",\n'
        '    "must_have": ["string"],\n'
        '    "nice_to_have": ["string"],\n'
        '    "avoid_now": ["string"]\n'
        "  },\n"
        '  "hard_constraints": {\n'
        '    "city": "string",\n'
        '    "open_now": true,\n'
        '    "max_distance_km": 0\n'
        "  }\n"
        "}"
    )


def profiler_user_prompt(payload: dict[str, Any]) -> str:
    return "Input JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def forecaster_system_prompt() -> str:
    return (
        "You are ForecasterAgent in a CoMaPOI-styled multi-agent recommendation system.\n"
        "Goal: refine initial retrieval candidates into:\n"
        "- long-term candidate set CH (profile-aligned)\n"
        "- short-term candidate set CC (real-time intent aligned)\n"
        "Rules:\n"
        "- business_id must come from provided candidates only.\n"
        "- Integrate user_profile_features if available.\n"
        "- Keep candidate lists focused and high precision.\n"
        "- Output valid JSON only.\n"
        "JSON schema:\n"
        "{\n"
        '  "long_term_candidate_ids": ["business_id"],\n'
        '  "short_term_candidate_ids": ["business_id"],\n'
        '  "merged_candidate_ids": ["business_id"],\n'
        '  "rerank_rationale": ["short bullet string"]\n'
        "}"
    )


def forecaster_user_prompt(payload: dict[str, Any]) -> str:
    return "Input JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def predictor_system_prompt() -> str:
    return (
        "You are PredictorAgent in a CoMaPOI-styled multi-agent recommendation system.\n"
        "Goal: produce final Top-N POI recommendations for a query-based real-time task.\n"
        "You must integrate: long-term profile, short-term pattern, CH, CC, and candidate facts.\n"
        "Rules:\n"
        "- Pick business_id only from provided final candidates.\n"
        "- Use user_profile_features as prior preference evidence when available.\n"
        "- Rank by practical suitability for the given time/location/query.\n"
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


def predictor_user_prompt(payload: dict[str, Any]) -> str:
    return "Input JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
