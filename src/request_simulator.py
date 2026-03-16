from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random

from src.schemas import GeoPoint, UserQueryContext


@dataclass
class QueryTemplate:
    text: str
    long_term_notes: str = ""
    recent_activity_notes: str = ""


class RequestSimulator:
    def __init__(self, seed: int = 7) -> None:
        self.random = Random(seed)

    def sample(
        self,
        templates: list[QueryTemplate],
        base_time: datetime,
        city: str,
        state: str,
        center_lat: float,
        center_lon: float,
        user_id: str = "sim_user",
    ) -> UserQueryContext:
        template = self.random.choice(templates)
        minute_offset = self.random.randint(-180, 180)
        sampled_time = base_time + timedelta(minutes=minute_offset)
        lat_jitter = self.random.uniform(-0.03, 0.03)
        lon_jitter = self.random.uniform(-0.03, 0.03)

        return UserQueryContext(
            query_text=template.text,
            local_time=sampled_time.isoformat(timespec="minutes"),
            location=GeoPoint(lat=center_lat + lat_jitter, lon=center_lon + lon_jitter),
            city=city,
            state=state,
            user_id=user_id,
            long_term_notes=template.long_term_notes,
            recent_activity_notes=template.recent_activity_notes,
        )

