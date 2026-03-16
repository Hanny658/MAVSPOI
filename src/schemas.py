from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GeoPoint:
    lat: float
    lon: float


@dataclass
class UserQueryContext:
    query_text: str
    local_time: str
    location: GeoPoint | None = None
    city: str = ""
    state: str = ""
    user_id: str = "anonymous"
    long_term_notes: str = ""
    recent_activity_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.location is None:
            data["location"] = None
        return data


@dataclass
class BusinessPOI:
    business_id: str
    name: str
    address: str
    city: str
    state: str
    latitude: float
    longitude: float
    categories: list[str] = field(default_factory=list)
    stars: float = 0.0
    review_count: int = 0
    is_open: int = 1
    raw_hours: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def category_text(self) -> str:
        return ", ".join(self.categories)

    def to_retrieval_text(self) -> str:
        return (
            f"{self.name}. Category: {self.category_text}. "
            f"Located in {self.city}, {self.state}. "
            f"Stars {self.stars}. Reviews {self.review_count}. "
            f"Address: {self.address}."
        )

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "business_id": self.business_id,
            "name": self.name,
            "city": self.city,
            "state": self.state,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "categories": self.categories,
            "stars": self.stars,
            "review_count": self.review_count,
            "is_open": self.is_open,
        }


@dataclass
class CandidateScore:
    business: BusinessPOI
    score: float
    text_similarity: float
    geo_score: float
    popularity_score: float
    distance_km: float | None = None

    def to_compact_dict(self) -> dict[str, Any]:
        payload = {
            "business_id": self.business.business_id,
            "name": self.business.name,
            "city": self.business.city,
            "categories": self.business.categories,
            "stars": self.business.stars,
            "review_count": self.business.review_count,
            "score": round(self.score, 6),
            "text_similarity": round(self.text_similarity, 6),
            "geo_score": round(self.geo_score, 6),
            "popularity_score": round(self.popularity_score, 6),
        }
        if self.distance_km is not None:
            payload["distance_km"] = round(self.distance_km, 3)
        return payload

