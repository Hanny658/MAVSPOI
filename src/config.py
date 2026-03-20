from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> None:
        return None

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class Settings:
    openai_api_key: str
    openai_model: str
    openai_embed_model: str
    yelp_business_json: str
    yelp_profile_json: str
    yelp_max_businesses: int
    yelp_city_filter: str
    retrieval_top_k: int
    forecaster_top_k: int
    final_top_k: int
    embed_cache_path: str


DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULTS = {
    "yelp_business_json": "data/train/yelp-indianapolis-train-business.jsonl",
    "yelp_profile_json": "data/train/yelp-indianapolis-train-profile.jsonl",
    "yelp_max_businesses": 50000,
    "yelp_city_filter": "",
    "retrieval_top_k": 80,
    "forecaster_top_k": 25,
    "final_top_k": 10,
    "embed_cache_path": "data/cache/yelp-indianapolis-train-business-embeddings.faiss",
}

DEFAULT_MAVSPOI = {
    "router": {
        "enabled_agents": ["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
        "min_agents": 3,
        "max_agents": 4,
        "activation_threshold": 0.5,
        "fallback_agents": ["A1", "A3", "A4", "A6"],
        "default_max_distance_km": 10.0,
        "use_llm": True,
        "llm_temperature": 0.1,
        "hybrid_heuristic_base": 0.45,
        "hybrid_llm_base": 0.45,
        "hybrid_prior_base": 0.10,
    },
    "voting": {
        "candidate_pool_size": 30,
        "neutral_score": 0.5,
        "parallel_workers": 7,
        "llm_enabled": True,
        "llm_temperature": 0.1,
        "llm_max_tokens": 1600,
        "llm_candidate_limit": 30,
        "llm_weight": 0.5,
        "heuristic_weight": 0.5,
    },
    "aggregator": {
        "retrieval_weight": 0.34,
        "diversity_penalty": 0.005,
        "neutral_score": 0.5,
        "weights": {
            "A1": 0.16,
            "A2": 0.14,
            "A3": 0.2,
            "A4": 0.17,
            "A5": 0.09,
            "A6": 0.15,
            "A7": 0.09,
        },
    },
    "constraints": {
        "open_now": {
            "mode": "soft",
            "closed_penalty": 0.12,
        },
        "max_distance": {
            "mode": "soft",
            "distance_buffer_km": 0.0,
            "per_km_penalty": 0.03,
            "max_penalty": 0.25,
        },
    },
}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is not None:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid YAML root in {path}: expected object/dict.")
        runtime = raw.get("runtime", raw)
        if not isinstance(runtime, dict):
            raise ValueError(
                f"Invalid runtime section in {path}: expected object/dict."
            )
        return runtime
    return _parse_simple_yaml(path)


def _load_yaml_root(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is not None:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if isinstance(raw, dict):
            return raw
        return {}
    # Fallback parser only supports runtime keys; keep empty root in this path.
    return {}


def _parse_simple_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        return text


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    # Lightweight fallback parser for simple key-value YAML used in this repo.
    runtime: dict[str, Any] = {}
    in_runtime = False
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("runtime:"):
                in_runtime = True
                continue
            if not in_runtime and line.startswith(" "):
                # Ignore nested sections before runtime.
                continue
            if in_runtime and not line.startswith(" "):
                # Exit runtime block when a new top-level key begins.
                in_runtime = False
            if ":" not in stripped:
                continue

            # Parse top-level keys only when runtime block is absent.
            if in_runtime:
                target = stripped
            elif line.startswith(" "):
                continue
            else:
                target = stripped

            key, value = target.split(":", 1)
            key = key.strip()
            if not key:
                continue
            runtime[key] = _parse_simple_scalar(value)
    return runtime


def _pick(
    yaml_cfg: dict[str, Any],
    yaml_key: str,
    env_key: str,
    default: Any,
) -> Any:
    # Primary source: YAML. Env override is kept for backward compatibility.
    if yaml_key in yaml_cfg and yaml_cfg[yaml_key] is not None:
        base = yaml_cfg[yaml_key]
    else:
        base = default
    env_val = os.getenv(env_key, "").strip()
    if env_val:
        return env_val
    return base


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = set(base.keys()) | set(override.keys())
    for key in keys:
        b_val = base.get(key)
        o_val = override.get(key)
        if isinstance(b_val, dict) and isinstance(o_val, dict):
            out[key] = _deep_merge(b_val, o_val)
        elif key in override:
            out[key] = o_val
        else:
            out[key] = b_val
    return out


def load_mavspoi_config(config_path: str | None = None) -> dict[str, Any]:
    load_dotenv()
    raw_path = config_path or os.getenv("CONFIG_YAML_PATH", DEFAULT_CONFIG_PATH).strip() or DEFAULT_CONFIG_PATH
    root = _load_yaml_root(Path(raw_path))
    mav_raw = root.get("mavspoi", {})
    if not isinstance(mav_raw, dict):
        mav_raw = {}
    return _deep_merge(DEFAULT_MAVSPOI, mav_raw)


def load_settings() -> Settings:
    load_dotenv()
    yaml_path = Path(os.getenv("CONFIG_YAML_PATH", DEFAULT_CONFIG_PATH).strip() or DEFAULT_CONFIG_PATH)
    yaml_cfg = _load_yaml(yaml_path)

    yelp_business_json = _as_str(
        _pick(
            yaml_cfg,
            "yelp_business_json",
            "YELP_BUSINESS_JSON",
            DEFAULTS["yelp_business_json"],
        ),
        str(DEFAULTS["yelp_business_json"]),
    )
    yelp_profile_json = _as_str(
        _pick(
            yaml_cfg,
            "yelp_profile_json",
            "YELP_PROFILE_JSON",
            DEFAULTS["yelp_profile_json"],
        ),
        str(DEFAULTS["yelp_profile_json"]),
    )
    yelp_max_businesses = _as_int(
        _pick(
            yaml_cfg,
            "yelp_max_businesses",
            "YELP_MAX_BUSINESSES",
            DEFAULTS["yelp_max_businesses"],
        ),
        int(DEFAULTS["yelp_max_businesses"]),
    )
    yelp_city_filter = _as_str(
        _pick(
            yaml_cfg,
            "yelp_city_filter",
            "YELP_CITY_FILTER",
            DEFAULTS["yelp_city_filter"],
        ),
        str(DEFAULTS["yelp_city_filter"]),
    )
    retrieval_top_k = _as_int(
        _pick(
            yaml_cfg,
            "retrieval_top_k",
            "RETRIEVAL_TOP_K",
            DEFAULTS["retrieval_top_k"],
        ),
        int(DEFAULTS["retrieval_top_k"]),
    )
    forecaster_top_k = _as_int(
        _pick(
            yaml_cfg,
            "forecaster_top_k",
            "FORECASTER_TOP_K",
            DEFAULTS["forecaster_top_k"],
        ),
        int(DEFAULTS["forecaster_top_k"]),
    )
    final_top_k = _as_int(
        _pick(
            yaml_cfg,
            "final_top_k",
            "FINAL_TOP_K",
            DEFAULTS["final_top_k"],
        ),
        int(DEFAULTS["final_top_k"]),
    )
    embed_cache_path = _as_str(
        _pick(
            yaml_cfg,
            "embed_cache_path",
            "EMBED_CACHE_PATH",
            DEFAULTS["embed_cache_path"],
        ),
        str(DEFAULTS["embed_cache_path"]),
    )

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
        openai_embed_model=os.getenv(
            "OPENAI_EMBED_MODEL", "text-embedding-3-small"
        ).strip(),
        yelp_business_json=yelp_business_json,
        yelp_profile_json=yelp_profile_json,
        yelp_max_businesses=yelp_max_businesses,
        yelp_city_filter=yelp_city_filter,
        retrieval_top_k=retrieval_top_k,
        forecaster_top_k=forecaster_top_k,
        final_top_k=final_top_k,
        embed_cache_path=embed_cache_path,
    )
