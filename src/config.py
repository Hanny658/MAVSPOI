from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


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


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    return int(value)


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
        openai_embed_model=os.getenv(
            "OPENAI_EMBED_MODEL", "text-embedding-3-small"
        ).strip(),
        yelp_business_json=os.getenv(
            "YELP_BUSINESS_JSON", "data/train/yelp-indianapolis-train-business.jsonl"
        ).strip(),
        yelp_profile_json=os.getenv(
            "YELP_PROFILE_JSON", "data/train/yelp-indianapolis-train-profile.jsonl"
        ).strip(),
        yelp_max_businesses=_get_int("YELP_MAX_BUSINESSES", 50000),
        yelp_city_filter=os.getenv("YELP_CITY_FILTER", "").strip(),
        retrieval_top_k=_get_int("RETRIEVAL_TOP_K", 80),
        forecaster_top_k=_get_int("FORECASTER_TOP_K", 25),
        final_top_k=_get_int("FINAL_TOP_K", 10),
        embed_cache_path=os.getenv(
            "EMBED_CACHE_PATH", "data/cache/yelp-indianapolis-train-business-embeddings.jsonl"
        ).strip(),
    )
