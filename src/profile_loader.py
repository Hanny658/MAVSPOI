from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_user_profiles(profile_path: str) -> dict[str, dict[str, Any]]:
    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    profiles: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        user_id = str(row.get("user_id", "")).strip()
        if not user_id:
            continue
        profiles[user_id] = row
    if not profiles:
        raise ValueError(f"No user profiles loaded from: {path}")
    return profiles

