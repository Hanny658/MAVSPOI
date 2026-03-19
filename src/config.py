from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

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
