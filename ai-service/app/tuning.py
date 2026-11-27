"""Tuning values that real conversation data should decide (plan §07, §08, §06).

Every value here is a *default*, chosen so the service behaves sensibly before any
measurement exists. They are grouped by the data that should eventually decide them
(see docs/tuning.md). Override them without a code change:

* a JSON file: ``AI_TUNING_FILE=/path/tuning.json`` (partial: only the keys you change)
* one env var per value: ``AI_TUNING__RETRIEVAL__W_SEMANTIC=0.6``
* the whole block as JSON: ``AI_TUNING='{"relationship": {"daily_positive_cap": 8}}'``

Env values win over the file. ``python -m tools.show_tuning`` prints the effective values.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _T(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievalTuning(_T):
    """Hybrid memory score (retriever.py). Decided by: recall@k / precision on the memory eval set."""

    w_semantic: float = Field(0.55, ge=0)
    w_keyword: float = Field(0.15, ge=0)
    w_importance: float = Field(0.15, ge=0)
    w_time: float = Field(0.15, ge=0)
    recency_decay_days: float = Field(30.0, gt=0)  # general memories: exp(-age / N)
    schedule_full_days: float = Field(2.0, ge=0)  # schedules within N days of now score 1.0
    schedule_fade_days: float = Field(12.0, gt=0)  # then fade linearly to 0 over N days
    candidate_limit: int = Field(500, ge=1)  # newest N valid memories scored per query


class UnfinishedTuning(_T):
    """Open promises/plans exposure (unfinished.py). Decided by: how often a follow-up feels natural."""

    due_window_hours: float = Field(24.0, ge=0)
    expire_after_days: float = Field(14.0, gt=0)  # dated events
    undated_expire_after_days: float = Field(28.0, gt=0)
    max_exposed: int = Field(2, ge=0)


DEFAULT_DELTAS: dict[str, tuple[float, float, float]] = {
    # kind -> (trust, closeness, tension)
    "PERSONAL_DISCLOSURE": (4, 2, 0),
    "PROMISE_KEPT": (5, 0, 0),
    "PROMISE_BROKEN": (-4, 0, 3),
    "CONFLICT": (0, 0, 6),
    "RECONCILIATION": (0, 3, -8),
    "SHARED_EXPERIENCE": (0, 3, 0),
    "SUPPORT": (0, 2, 0),
}


class RelationshipTuning(_T):
    """Relationship Reducer (relationship/reducer.py). Decided by: stage-change pacing in long scenarios."""

    deltas: dict[str, tuple[float, float, float]] = Field(default_factory=lambda: dict(DEFAULT_DELTAS))
    daily_positive_cap: float = Field(6.0, ge=0)  # SHARED_EXPERIENCE / SUPPORT closeness per local day
    hysteresis: float = Field(3.0, ge=0)
    tense_tension: float = 50.0
    close_closeness: float = 60.0
    close_trust: float = 55.0
    familiar_closeness: float = 35.0
    default_trust: float = 50.0  # used until (or if) the baseline model call succeeds
    default_closeness: float = 40.0
    default_tension: float = 0.0

    @field_validator("deltas", mode="before")
    @classmethod
    def _merge_deltas(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        v = {str(k).upper(): d for k, d in v.items()}  # env keys arrive lower-cased
        unknown = set(v) - set(DEFAULT_DELTAS)
        if unknown:
            raise ValueError(f"unknown relation kinds: {sorted(unknown)}")
        return {**DEFAULT_DELTAS, **v}


DEFAULT_BUDGETS: dict[str, int] = {
    "policy": 400,
    "character": 1200,
    "persona": 400,
    "relationship": 500,
    "memories": 900,
    "summary": 800,
    "time": 60,
    "recent": 3000,
}


class ContextTuning(_T):
    """Context Builder layer budgets in tokens (context_builder.py). Decided by: model context size,
    measured layer usage (``layerTokens`` in snapshots) and quality at the trim points."""

    budgets: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_BUDGETS))
    output_tokens: int = Field(600, ge=16)
    chat_temperature: float = Field(0.8, ge=0, le=2)

    @field_validator("budgets", mode="before")
    @classmethod
    def _merge_budgets(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        unknown = set(v) - set(DEFAULT_BUDGETS)
        if unknown:
            raise ValueError(f"unknown context layers: {sorted(unknown)}")
        return {**DEFAULT_BUDGETS, **v}

    @property
    def input_budget(self) -> int:
        return sum(self.budgets.values())


class MemoryTuning(_T):
    """Memory pipeline (memory/*.py). Decided by: extraction precision and cost per turn."""

    max_new_per_turn: int = Field(5, ge=0)
    prefilter_min_chars: int = Field(12, ge=0)  # free-form lines shorter than this skip extraction
    max_runs: int = Field(3, ge=1)  # provider retries before a memory job gives up
    reindex_batch: int = Field(16, ge=0)  # stale/failed embeddings retried per memory job
    revision_check_batch: int = Field(40, ge=1)  # memories judged per call after a character edit


class Tuning(_T):
    retrieval: RetrievalTuning = Field(default_factory=RetrievalTuning)
    unfinished: UnfinishedTuning = Field(default_factory=UnfinishedTuning)
    relationship: RelationshipTuning = Field(default_factory=RelationshipTuning)
    context: ContextTuning = Field(default_factory=ContextTuning)
    memory: MemoryTuning = Field(default_factory=MemoryTuning)

    @model_validator(mode="after")
    def _weights_positive(self) -> Tuning:
        r = self.retrieval
        if r.w_semantic + r.w_keyword + r.w_importance + r.w_time <= 0:
            raise ValueError("retrieval weights must not all be zero")
        return self

    def version(self) -> str:
        """Short hash of the effective values; recorded with snapshots so evals can tell runs apart."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return "tuning-" + hashlib.sha256(blob.encode()).hexdigest()[:10]


DEFAULT_TUNING = Tuning()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
