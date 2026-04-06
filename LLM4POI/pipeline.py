from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.config import Settings
from src.openai_client import OpenAIService
from src.profile_loader import load_user_profiles
from src.retrieval import CandidateRetriever
from src.schemas import CandidateScore, UserQueryContext
from src.yelp_loader import load_yelp_businesses


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_dt(text: str) -> datetime | None:
    text = str(text or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _derive_sibling(path: Path, src_key: str, dst_key: str) -> Path:
    name = path.name
    if src_key in name:
        return path.with_name(name.replace(src_key, dst_key))
    return path.with_name(name.replace("business", dst_key))


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _profile_top_categories(profile: dict[str, Any], limit: int = 5) -> list[str]:
    category_rows = profile.get("category_pref", {}).get("top_categories", [])
    if not isinstance(category_rows, list):
        return []
    output: list[str] = []
    for row in category_rows:
        if not isinstance(row, dict):
            continue
        cat = str(row.get("category", "")).strip()
        if cat:
            output.append(cat)
        if len(output) >= limit:
            break
    return output


def _compact_profile_for_model(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {}
    support = profile.get("support", {})
    static = profile.get("static", {})
    temporal_pref = profile.get("temporal_pref", {})
    price_pref = profile.get("price_pref", {})
    spatial_pref = profile.get("spatial_pref", {})

    return {
        "profile_version": profile.get("profile_version", ""),
        "support_level": support.get("support_level", "unknown"),
        "interaction_evidence_count": support.get("interaction_evidence_count", 0),
        "top_categories": _profile_top_categories(profile, limit=8),
        "dominant_price_level": price_pref.get("dominant_price_level", ""),
        "active_hours_top3": temporal_pref.get("active_hours_top3", []),
        "weekend_ratio": temporal_pref.get("weekend_ratio", 0.0),
        "radius_km_p50": spatial_pref.get("radius_km_p50", None),
        "radius_km_p90": spatial_pref.get("radius_km_p90", None),
        "user_average_stars": static.get("user_average_stars", 0.0),
        "fans": static.get("fans", 0),
        "friend_count": static.get("friend_count", 0),
    }


def _profile_to_long_term_notes(profile: dict[str, Any] | None) -> str:
    if not profile:
        return ""
    support = profile.get("support", {})
    spatial = profile.get("spatial_pref", {})
    business_quality = profile.get("business_quality_pref", {})
    price_pref = profile.get("price_pref", {})

    pieces: list[str] = []
    pieces.append(f"support_level={support.get('support_level', 'unknown')}")
    pieces.append(f"interaction_count={support.get('interaction_evidence_count', 0)}")

    cats = _profile_top_categories(profile, limit=5)
    if cats:
        pieces.append("top_categories=" + ", ".join(cats))

    dominant_price = str(price_pref.get("dominant_price_level", "")).strip()
    if dominant_price:
        pieces.append(f"price_level={dominant_price}")

    if spatial.get("radius_km_p50") is not None:
        pieces.append(f"radius_p50_km={spatial.get('radius_km_p50')}")
    if spatial.get("radius_km_p90") is not None:
        pieces.append(f"radius_p90_km={spatial.get('radius_km_p90')}")
    if business_quality.get("avg_business_stars") is not None:
        pieces.append(
            f"avg_visited_business_stars={business_quality.get('avg_business_stars')}"
        )
    return "Profile-derived long-term signals: " + "; ".join(pieces)


def _profile_to_recent_notes(profile: dict[str, Any] | None) -> str:
    if not profile:
        return ""
    temporal = profile.get("temporal_pref", {})
    checkin = profile.get("checkin_context_pref", {})
    pieces: list[str] = []
    active_hours = temporal.get("active_hours_top3", [])
    if isinstance(active_hours, list) and active_hours:
        pieces.append("active_hours_top3=" + ",".join(str(x) for x in active_hours))
    weekend_ratio = temporal.get("weekend_ratio")
    if weekend_ratio is not None:
        pieces.append(f"weekend_ratio={weekend_ratio}")
    avg_checkins = checkin.get("avg_business_checkins")
    if avg_checkins is not None:
        pieces.append(f"avg_business_checkins={avg_checkins}")
    if not pieces:
        return ""
    return "Profile-derived short-term hints: " + "; ".join(pieces)


def _merge_notes(manual_note: str, profile_note: str) -> str:
    manual_note = manual_note.strip()
    profile_note = profile_note.strip()
    if manual_note and profile_note:
        return manual_note + " | " + profile_note
    return manual_note or profile_note


@dataclass(frozen=True)
class CheckinEvent:
    user_id: str
    business_id: str
    timestamp: datetime
    date_text: str
    category_name: str
    category_id: int


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    user_id: str
    start_time: datetime
    end_time: datetime
    events: tuple[CheckinEvent, ...]


class LLM4POIRealtimeRecommender:
    def __init__(
        self,
        settings: Settings,
        variant: str = "llm4poi",
        trajectory_gap_hours: int = 24,
        history_top_trajectories: int = 20,
        max_history_records: int = 300,
        max_current_records: int = 50,
        max_similarity_pool: int = 1200,
        max_query_trajectories: int = 6000,
        rerank_pool_size: int | None = None,
        few_shot_examples: int = 3,
        llm_stage1_temperature: float = 0.0,
        llm_stage2_temperature: float = 0.1,
        llm_retrieval_blend: float = 0.25,
    ) -> None:
        self.settings = settings
        self.variant = variant.strip().lower()
        self.trajectory_gap_hours = max(1, int(trajectory_gap_hours))
        self.history_top_trajectories = max(0, int(history_top_trajectories))
        self.max_history_records = max(0, int(max_history_records))
        self.max_current_records = max(1, int(max_current_records))
        self.max_similarity_pool = max(1, int(max_similarity_pool))
        self.max_query_trajectories = max(1, int(max_query_trajectories))
        self.rerank_pool_size = rerank_pool_size
        self.few_shot_examples = max(0, int(few_shot_examples))
        self.llm_stage1_temperature = _clip01(llm_stage1_temperature)
        self.llm_stage2_temperature = _clip01(llm_stage2_temperature)
        self.llm_retrieval_blend = _clip01(llm_retrieval_blend)

        self.use_history_block = self.variant in {"llm4poi", "llm4poi_star2"}
        self.use_similarity = self.variant == "llm4poi"
        self.include_other_users = self.variant == "llm4poi"
        if self.variant == "llm4poi_star":
            self.use_history_block = False
            self.use_similarity = False
            self.include_other_users = False

        self.openai_service = OpenAIService(settings)
        self.businesses = load_yelp_businesses(
            business_json_path=settings.yelp_business_json,
            max_businesses=settings.yelp_max_businesses,
            city_filter=settings.yelp_city_filter,
        )
        self.user_profiles = load_user_profiles(settings.yelp_profile_json)
        self.retriever = CandidateRetriever(
            openai_service=self.openai_service,
            businesses=self.businesses,
            embed_cache_path=settings.embed_cache_path,
        )

        self.business_by_id = {b.business_id: b for b in self.businesses}
        self.category_to_id = self._build_category_index()
        self.user_events, self.trajectories = self._load_user_events_and_trajectories()
        self.trajectory_by_id = {t.trajectory_id: t for t in self.trajectories}
        self.query_trajectory_ids, self.query_trajectory_matrix = (
            self._build_query_trajectory_embeddings()
        )
        self.query_traj_index_map = {
            tid: idx for idx, tid in enumerate(self.query_trajectory_ids)
        }

    def _build_category_index(self) -> dict[str, int]:
        cats: list[str] = []
        for b in self.businesses:
            for cat in b.categories:
                text = str(cat).strip()
                if text:
                    cats.append(text)
        ordered = sorted(set(cats))
        return {cat: i for i, cat in enumerate(ordered)}

    def _business_category(self, business_id: str) -> tuple[str, int]:
        poi = self.business_by_id.get(business_id)
        if poi is None or not poi.categories:
            return "unknown", -1
        category_name = str(poi.categories[0]).strip() or "unknown"
        return category_name, _safe_int(self.category_to_id.get(category_name, -1), -1)

    def _resolve_train_paths(self) -> tuple[Path, Path]:
        business_path = Path(self.settings.yelp_business_json)
        review_path = _derive_sibling(business_path, "-business", "-review")
        tip_path = _derive_sibling(business_path, "-business", "-tip")
        return review_path, tip_path

    def _load_user_events_and_trajectories(
        self,
    ) -> tuple[dict[str, list[CheckinEvent]], list[Trajectory]]:
        review_path, tip_path = self._resolve_train_paths()
        events_by_user: dict[str, list[CheckinEvent]] = {}
        valid_ids = set(self.business_by_id.keys())

        def _append_event(user_id: str, business_id: str, date_text: str) -> None:
            if user_id not in events_by_user:
                events_by_user[user_id] = []
            category_name, category_id = self._business_category(business_id)
            dt = _parse_dt(date_text)
            if dt is None:
                return
            events_by_user[user_id].append(
                CheckinEvent(
                    user_id=user_id,
                    business_id=business_id,
                    timestamp=dt,
                    date_text=date_text,
                    category_name=category_name,
                    category_id=category_id,
                )
            )

        if review_path.exists():
            for row in _iter_jsonl(review_path):
                user_id = str(row.get("user_id", "")).strip()
                business_id = str(row.get("business_id", "")).strip()
                date_text = str(row.get("date", "")).strip()
                if (
                    not user_id
                    or not business_id
                    or not date_text
                    or business_id not in valid_ids
                ):
                    continue
                _append_event(user_id=user_id, business_id=business_id, date_text=date_text)

        if tip_path.exists():
            for row in _iter_jsonl(tip_path):
                user_id = str(row.get("user_id", "")).strip()
                business_id = str(row.get("business_id", "")).strip()
                date_text = str(row.get("date", "")).strip()
                if (
                    not user_id
                    or not business_id
                    or not date_text
                    or business_id not in valid_ids
                ):
                    continue
                _append_event(user_id=user_id, business_id=business_id, date_text=date_text)

        for user_id in list(events_by_user.keys()):
            events_by_user[user_id].sort(key=lambda x: x.timestamp)

        trajectories: list[Trajectory] = []
        gap = timedelta(hours=self.trajectory_gap_hours)
        for user_id, events in events_by_user.items():
            if not events:
                continue
            current: list[CheckinEvent] = [events[0]]
            idx = 0
            for event in events[1:]:
                prev = current[-1]
                if event.timestamp - prev.timestamp > gap:
                    if len(current) > 1:
                        idx += 1
                        trajectories.append(
                            Trajectory(
                                trajectory_id=f"{user_id}-{idx}",
                                user_id=user_id,
                                start_time=current[0].timestamp,
                                end_time=current[-1].timestamp,
                                events=tuple(current),
                            )
                        )
                    current = [event]
                else:
                    current.append(event)
            if len(current) > 1:
                idx += 1
                trajectories.append(
                    Trajectory(
                        trajectory_id=f"{user_id}-{idx}",
                        user_id=user_id,
                        start_time=current[0].timestamp,
                        end_time=current[-1].timestamp,
                        events=tuple(current),
                    )
                )

        trajectories.sort(key=lambda t: t.end_time)
        return events_by_user, trajectories

    def _event_to_sentence(self, event: CheckinEvent) -> str:
        return (
            f"At {event.date_text}, user {event.user_id} visited POI id {event.business_id} "
            f"which is a/an {event.category_name} with category id {event.category_id}."
        )

    def _trajectory_to_query_prompt(self, trajectory: Trajectory) -> str:
        rows = [self._event_to_sentence(e) for e in trajectory.events]
        return f"Trajectory for user {trajectory.user_id}: " + " ".join(rows)

    def _trajectory_to_key_prompt(self, events: list[CheckinEvent], user_id: str) -> str:
        key_events = list(events[:-1]) if len(events) >= 2 else list(events)
        if not key_events:
            key_events = list(events)
        rows = [self._event_to_sentence(e) for e in key_events]
        return f"Trajectory for user {user_id}: " + " ".join(rows)

    def _embed_texts_normalized(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors: list[list[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors.extend(self.openai_service.embed_texts(batch))
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def _build_query_trajectory_embeddings(self) -> tuple[list[str], np.ndarray]:
        if not self.trajectories or not self.use_history_block:
            return [], np.zeros((0, 1), dtype=np.float32)
        selected = sorted(
            self.trajectories,
            key=lambda x: x.end_time,
            reverse=True,
        )[: self.max_query_trajectories]
        selected = list(reversed(selected))
        ids = [t.trajectory_id for t in selected]
        if not self.use_similarity:
            return ids, np.zeros((0, 1), dtype=np.float32)
        texts = [self._trajectory_to_query_prompt(t) for t in selected]
        matrix = self._embed_texts_normalized(texts)
        return ids, matrix

    def _build_enriched_context(
        self, context: UserQueryContext
    ) -> tuple[UserQueryContext, dict[str, Any]]:
        profile = self.user_profiles.get(context.user_id)
        profile_long = _profile_to_long_term_notes(profile)
        profile_recent = _profile_to_recent_notes(profile)
        enriched = UserQueryContext(
            query_text=context.query_text,
            local_time=context.local_time,
            location=context.location,
            city=context.city,
            state=context.state,
            user_id=context.user_id,
            long_term_notes=_merge_notes(context.long_term_notes, profile_long),
            recent_activity_notes=_merge_notes(context.recent_activity_notes, profile_recent),
        )
        return enriched, _compact_profile_for_model(profile)

    def _target_rerank_pool_size(self, top_k: int) -> int:
        base = self.rerank_pool_size or self.settings.forecaster_top_k
        return max(top_k, int(base))

    def _select_current_events(
        self,
        context: UserQueryContext,
    ) -> tuple[list[CheckinEvent], datetime]:
        query_time = _parse_dt(context.local_time) or datetime.now()
        events = self.user_events.get(context.user_id, [])
        if not events:
            return [], query_time
        past = [e for e in events if e.timestamp <= query_time]
        if not past:
            return [], query_time
        tail: list[CheckinEvent] = []
        end_time = past[-1].timestamp
        gap = timedelta(hours=self.trajectory_gap_hours)
        for event in reversed(past):
            if end_time - event.timestamp <= gap:
                tail.append(event)
            else:
                break
        tail.reverse()
        if not tail:
            tail = [past[-1]]
        tail = tail[-self.max_current_records :]
        return tail, query_time

    def _eligible_trajectories(
        self,
        current_user_id: str,
        current_start_time: datetime,
        query_time: datetime,
    ) -> list[Trajectory]:
        rows = [t for t in self.trajectories if t.end_time < current_start_time]
        if not rows:
            rows = [t for t in self.trajectories if t.end_time < query_time]
        if not self.include_other_users:
            rows = [t for t in rows if t.user_id == current_user_id]
        rows.sort(key=lambda x: x.end_time, reverse=True)
        return rows[: self.max_similarity_pool]

    def _select_history_trajectories(
        self,
        context: UserQueryContext,
        current_events: list[CheckinEvent],
        query_time: datetime,
    ) -> tuple[list[Trajectory], list[dict[str, Any]]]:
        if (
            not self.use_history_block
            or self.history_top_trajectories <= 0
            or not current_events
        ):
            return [], []

        current_start_time = current_events[0].timestamp
        eligible = self._eligible_trajectories(
            current_user_id=context.user_id,
            current_start_time=current_start_time,
            query_time=query_time,
        )
        if not eligible:
            return [], []

        if not self.use_similarity:
            picked = eligible[: self.history_top_trajectories]
            meta = [
                {
                    "trajectory_id": t.trajectory_id,
                    "similarity": None,
                    "user_id": t.user_id,
                    "event_count": len(t.events),
                }
                for t in picked
            ]
            return picked, meta

        key_text = self._trajectory_to_key_prompt(current_events, context.user_id)
        key_vec = self._embed_texts_normalized([key_text])
        if key_vec.shape[0] == 0:
            return [], []
        key = key_vec[0]

        candidate_ids = [t.trajectory_id for t in eligible]
        cached_ids: list[str] = []
        cached_vecs: list[np.ndarray] = []
        missing_ids: list[str] = []
        for tid in candidate_ids:
            idx = self.query_traj_index_map.get(tid)
            if idx is None:
                missing_ids.append(tid)
                continue
            cached_ids.append(tid)
            cached_vecs.append(self.query_trajectory_matrix[idx])

        if missing_ids:
            missing_texts = [
                self._trajectory_to_query_prompt(self.trajectory_by_id[tid])
                for tid in missing_ids
                if tid in self.trajectory_by_id
            ]
            if missing_texts:
                missing_matrix = self._embed_texts_normalized(missing_texts)
                for tid, vec in zip(missing_ids, missing_matrix):
                    cached_ids.append(tid)
                    cached_vecs.append(vec)

        if not cached_vecs:
            return [], []
        mat = np.vstack(cached_vecs).astype(np.float32)
        sims = mat @ key.astype(np.float32)
        order = np.argsort(-sims)

        picked: list[Trajectory] = []
        meta: list[dict[str, Any]] = []
        for idx in order[: self.history_top_trajectories]:
            tid = cached_ids[int(idx)]
            traj = self.trajectory_by_id.get(tid)
            if traj is None:
                continue
            picked.append(traj)
            meta.append(
                {
                    "trajectory_id": tid,
                    "similarity": round(float(sims[int(idx)]), 6),
                    "user_id": traj.user_id,
                    "event_count": len(traj.events),
                }
            )
        return picked, meta

    def _events_to_block_text(
        self,
        events: list[CheckinEvent],
        drop_last: bool,
    ) -> str:
        rows = [self._event_to_sentence(e) for e in events]
        if drop_last and len(rows) >= 2:
            rows = rows[:-1]
        return " ".join(rows)

    def _build_few_shot_examples(
        self,
        history_trajectories: list[Trajectory],
    ) -> list[dict[str, str]]:
        if self.few_shot_examples <= 0:
            return []
        examples: list[dict[str, str]] = []
        for traj in history_trajectories:
            if len(traj.events) < 2:
                continue
            obs = list(traj.events[:-1])
            target = traj.events[-1]
            question = (
                f"<question> The following is a trajectory of user {traj.user_id}: "
                f"{self._events_to_block_text(obs, drop_last=False)} "
                f"Given the data, at {target.date_text}, which POI id will user {traj.user_id} visit?"
            )
            answer = (
                f"<answer> At {target.date_text}, user {traj.user_id} will visit POI id "
                f"{target.business_id}."
            )
            examples.append({"question": question, "answer": answer})
            if len(examples) >= self.few_shot_examples:
                break
        return examples

    def _build_stage1_prompt(
        self,
        context: UserQueryContext,
        current_events: list[CheckinEvent],
        history_trajectories: list[Trajectory],
        candidates: list[CandidateScore],
    ) -> tuple[str, str]:
        current_text = self._events_to_block_text(current_events, drop_last=True)
        if not current_text:
            current_text = "No recent check-in records are available."

        history_rows: list[str] = []
        for traj in history_trajectories:
            history_rows.append(self._events_to_block_text(list(traj.events), drop_last=False))
            if sum(len(x) for x in history_rows) >= 12000:
                break
        history_text = " ".join(x for x in history_rows if x.strip())
        if not history_text:
            history_text = "No additional historical trajectories are available."

        candidate_ids = [c.business.business_id for c in candidates]
        few_shot = self._build_few_shot_examples(history_trajectories)
        question = (
            f"<question> The following is a trajectory of user {context.user_id}: {current_text} "
            f"There is also historical data: {history_text} "
            f"Given the data, at {context.local_time}, which POI id will user {context.user_id} visit next? "
            f"POI ids must be one of: {candidate_ids}."
        )
        payload = {
            "few_shot_examples": few_shot,
            "paper_style_question": question,
            "query_text": context.query_text,
            "query_city": context.city,
            "query_state": context.state,
            "candidate_ids": candidate_ids,
            "long_term_notes": context.long_term_notes,
            "recent_activity_notes": context.recent_activity_notes,
        }
        system_prompt = (
            "You are LLM4POI stage-1 predictor.\n"
            "Task: mimic next-POI QA behavior and output a single next POI id.\n"
            "Think internally using trajectory patterns and history similarity, but output JSON only.\n"
            "JSON schema:\n"
            "{\n"
            '  "next_poi_id": "string",\n'
            '  "confidence": 0,\n'
            '  "reasoning_tags": ["string"],\n'
            '  "qa_style_answer": "string"\n'
            "}\n"
            "Rules:\n"
            "- next_poi_id must be in candidate_ids.\n"
            "- confidence in [0,1].\n"
            "- qa_style_answer follows: At [time], user [id] will visit POI id [poi_id]."
        )
        user_prompt = "Input JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        return system_prompt, user_prompt

    def _build_prompt_blocks(
        self,
        context: UserQueryContext,
        current_events: list[CheckinEvent],
        history_trajectories: list[Trajectory],
        candidates: list[CandidateScore],
        top_k: int,
        stage1_draft: dict[str, Any],
    ) -> tuple[str, str]:
        current_text = self._events_to_block_text(current_events, drop_last=True)
        if not current_text:
            current_text = "No recent check-in records are available."

        history_rows: list[str] = []
        for traj in history_trajectories:
            for event in traj.events:
                history_rows.append(self._event_to_sentence(event))
                if len(history_rows) >= self.max_history_records:
                    break
            if len(history_rows) >= self.max_history_records:
                break
        if not history_rows:
            history_text = "No additional historical trajectories are available."
        else:
            history_text = " ".join(history_rows)

        candidate_rows: list[dict[str, Any]] = []
        for c in candidates:
            category = c.business.categories[0] if c.business.categories else "unknown"
            candidate_rows.append(
                {
                    "business_id": c.business.business_id,
                    "name": c.business.name,
                    "category": category,
                    "city": c.business.city,
                    "state": c.business.state,
                    "is_open": int(c.business.is_open),
                    "retrieval_score": round(float(c.score), 6),
                }
            )

        draft_id = str(stage1_draft.get("next_poi_id", "")).strip()
        draft_conf = _clip01(_safe_float(stage1_draft.get("confidence", 0.0), 0.0))
        draft_tags = stage1_draft.get("reasoning_tags", [])
        if not isinstance(draft_tags, list):
            draft_tags = []
        few_shot = self._build_few_shot_examples(history_trajectories)
        question = (
            f"<question> The following is a trajectory of user {context.user_id}: {current_text}\n"
            f"There is also historical data: {history_text}\n"
            f"Given the data, at {context.local_time}, which POI id will user {context.user_id} visit next?"
        )
        payload = {
            "question": question,
            "few_shot_examples": few_shot,
            "candidate_pois": candidate_rows,
            "query_text": context.query_text,
            "query_city": context.city,
            "query_state": context.state,
            "long_term_notes": context.long_term_notes,
            "recent_activity_notes": context.recent_activity_notes,
            "stage1_draft": {
                "next_poi_id": draft_id,
                "confidence": draft_conf,
                "reasoning_tags": draft_tags[:6],
            },
            "target_top_k": top_k,
        }

        system_prompt = (
            "You are LLM4POI stage-2 reranker.\n"
            "Goal: approximate the fine-tuned next-POI behavior through prompt-only inference.\n"
            "Use paper-style trajectory QA context + stage1 draft + candidate evidence.\n"
            "Think step-by-step internally but do not reveal hidden reasoning.\n"
            "Output JSON only with schema:\n"
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
            "}\n"
            "Rules:\n"
            "- Choose business_id only from candidate_pois.\n"
            "- Prioritize immediate next-visit plausibility (top1 first), then fill top-k.\n"
            "- Use stage1_draft as prior, not as hard constraint.\n"
            "- Higher score means higher confidence; score in [0,1].\n"
            "- Do not output duplicate business_id."
        )
        user_prompt = "Input JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        return system_prompt, user_prompt

    def _sanitize_recommendations(
        self,
        raw_rows: Any,
        candidates: list[CandidateScore],
        top_k: int,
    ) -> list[dict[str, Any]]:
        allowed = {c.business.business_id for c in candidates}
        by_id = {c.business.business_id: c for c in candidates}
        if not isinstance(raw_rows, list):
            raw_rows = []
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            business_id = str(row.get("business_id", "")).strip()
            if business_id not in allowed or business_id in seen:
                continue
            score = _clip01(_safe_float(row.get("score", 0.5), 0.5))
            reason = str(row.get("reason", "")).strip()
            tags_raw = row.get("fit_tags", [])
            tags = [str(x) for x in tags_raw] if isinstance(tags_raw, list) else []
            c = by_id[business_id]
            out.append(
                {
                    "business": c.business.to_compact_dict(),
                    "ranking_score": score,
                    "reason": reason,
                    "fit_tags": tags[:6],
                    "retrieval_components": {
                        "retrieval_score": c.score,
                        "text_similarity": c.text_similarity,
                        "geo_score": c.geo_score,
                        "popularity_score": c.popularity_score,
                        "distance_km": c.distance_km,
                    },
                }
            )
            seen.add(business_id)
            if len(out) >= top_k:
                break
        return out

    def _fallback_recommendations(
        self,
        candidates: list[CandidateScore],
        top_k: int,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for c in candidates[:top_k]:
            out.append(
                {
                    "business": c.business.to_compact_dict(),
                    "ranking_score": _clip01(c.score),
                    "reason": "Fallback to retrieval ranking.",
                    "fit_tags": ["retrieval_fallback"],
                    "retrieval_components": {
                        "retrieval_score": c.score,
                        "text_similarity": c.text_similarity,
                        "geo_score": c.geo_score,
                        "popularity_score": c.popularity_score,
                        "distance_km": c.distance_km,
                    },
                }
            )
        return out

    def _retrieval_norm_map(self, candidates: list[CandidateScore]) -> dict[str, float]:
        if not candidates:
            return {}
        vals = [float(c.score) for c in candidates]
        lo, hi = min(vals), max(vals)
        out: dict[str, float] = {}
        for c in candidates:
            bid = c.business.business_id
            if hi - lo <= 1e-9:
                out[bid] = 0.5
            else:
                out[bid] = _clip01((float(c.score) - lo) / (hi - lo))
        return out

    def _prompt_engineering_calibration(
        self,
        rows: list[dict[str, Any]],
        candidates: list[CandidateScore],
        stage1_draft_id: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        retrieval_norm = self._retrieval_norm_map(candidates)
        for row in rows:
            bid = str(row.get("business", {}).get("business_id", "")).strip()
            llm_score = _clip01(_safe_float(row.get("ranking_score", 0.5), 0.5))
            retr = retrieval_norm.get(bid, 0.5)
            score = (1.0 - self.llm_retrieval_blend) * llm_score + self.llm_retrieval_blend * retr
            if stage1_draft_id and bid == stage1_draft_id:
                score = _clip01(score + 0.05)
                tags = row.get("fit_tags", [])
                if isinstance(tags, list):
                    row["fit_tags"] = _dedup_keep_order([str(x) for x in tags] + ["stage1_prior"])[:6]
            row["ranking_score"] = _clip01(score)
        rows.sort(key=lambda x: float(x.get("ranking_score", 0.0)), reverse=True)
        return rows[:top_k]

    def recommend(self, context: UserQueryContext, top_k: int | None = None) -> dict[str, Any]:
        return self._recommend_impl(context=context, top_k=top_k, candidate_business_ids=None)

    def recommend_with_candidates(
        self,
        context: UserQueryContext,
        candidate_business_ids: list[str],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        return self._recommend_impl(
            context=context,
            top_k=top_k,
            candidate_business_ids=candidate_business_ids,
        )

    def _recommend_impl(
        self,
        context: UserQueryContext,
        top_k: int | None,
        candidate_business_ids: list[str] | None,
    ) -> dict[str, Any]:
        top_k = top_k or self.settings.final_top_k
        rerank_pool_k = self._target_rerank_pool_size(top_k)
        enriched_context, profile_features = self._build_enriched_context(context)

        if candidate_business_ids:
            initial_candidates = self.retriever.retrieve_from_pool(
                context=enriched_context,
                candidate_business_ids=candidate_business_ids,
                top_k=self.settings.retrieval_top_k,
            )
            retrieval_mode = "candidate_constrained"
        else:
            initial_candidates = self.retriever.retrieve(
                context=enriched_context,
                top_k=self.settings.retrieval_top_k,
            )
            retrieval_mode = "full_corpus"

        candidate_pool = initial_candidates[:rerank_pool_k]
        current_events, query_time = self._select_current_events(enriched_context)
        history_trajectories, history_meta = self._select_history_trajectories(
            context=enriched_context,
            current_events=current_events,
            query_time=query_time,
        )

        stage1_result: dict[str, Any] = {}
        stage1_error = ""
        stage1_system, stage1_user = self._build_stage1_prompt(
            context=enriched_context,
            current_events=current_events,
            history_trajectories=history_trajectories,
            candidates=candidate_pool,
        )
        try:
            stage1_result = self.openai_service.chat_json(
                system_prompt=stage1_system,
                user_prompt=stage1_user,
                temperature=self.llm_stage1_temperature,
                max_tokens=500,
            )
        except Exception as exc:
            stage1_error = f"{type(exc).__name__}: {exc}"

        stage1_draft_id = str(stage1_result.get("next_poi_id", "")).strip()
        if stage1_draft_id and stage1_draft_id not in {c.business.business_id for c in candidate_pool}:
            stage1_draft_id = ""

        system_prompt, user_prompt = self._build_prompt_blocks(
            context=enriched_context,
            current_events=current_events,
            history_trajectories=history_trajectories,
            candidates=candidate_pool,
            top_k=top_k,
            stage1_draft=stage1_result,
        )

        llm_result: dict[str, Any] = {}
        llm_error = ""
        try:
            llm_result = self.openai_service.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.llm_stage2_temperature,
                max_tokens=1400,
            )
        except Exception as exc:
            llm_error = f"{type(exc).__name__}: {exc}"

        recommendations = self._sanitize_recommendations(
            raw_rows=llm_result.get("recommendations", []),
            candidates=candidate_pool,
            top_k=top_k,
        )
        if not recommendations:
            recommendations = self._fallback_recommendations(candidate_pool, top_k)
        recommendations = self._prompt_engineering_calibration(
            rows=recommendations,
            candidates=candidate_pool,
            stage1_draft_id=stage1_draft_id,
            top_k=top_k,
        )

        final_summary = llm_result.get("final_summary", "")
        if not isinstance(final_summary, str) or not final_summary.strip():
            final_summary = (
                "LLM4POI-style trajectory prompting with key-query trajectory similarity "
                "and API LLM inference."
            )

        return {
            "task": "query_based_real_time_poi_recommendation",
            "system": "llm4poi_api_baseline",
            "variant": self.variant,
            "context": context.to_dict(),
            "enriched_context": enriched_context.to_dict(),
            "retrieval_mode": retrieval_mode,
            "user_profile_features": profile_features,
            "intermediate": {
                "retrieval_pool_size": len(initial_candidates),
                "rerank_pool_size": len(candidate_pool),
                "current_trajectory_records": len(current_events),
                "history_trajectory_count": len(history_trajectories),
                "history_record_budget": self.max_history_records,
                "history_trajectory_meta": history_meta[:20],
                "trajectory_pool_size": len(self.trajectories),
                "query_trajectory_embedding_count": len(self.query_trajectory_ids),
                "stage1_draft": stage1_result,
                "stage1_error": stage1_error,
                "llm_error": llm_error,
            },
            "recommendations": recommendations,
            "summary": final_summary,
        }
