"""AI DB schema (plan §11).

Every conversation-scoped table carries ``tenant_id, owner_id, conversation_id, epoch``.
Reads and writes always filter on that scope; the AI DB never joins BE tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONB, list[Any]: JSONB}


class ScopedMixin:
    tenant_id: Mapped[str] = mapped_column(String(64))
    owner_id: Mapped[str] = mapped_column(String(64))
    conversation_id: Mapped[str] = mapped_column(String(64))
    epoch: Mapped[int] = mapped_column(Integer)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------- conversation & lifecycle


class AiConversation(ScopedMixin, TimestampMixin, Base):
    __tablename__ = "ai_conversation"
    __table_args__ = (UniqueConstraint("tenant_id", "conversation_id", "epoch"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    character_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="ACTIVE")  # ACTIVE | REVOKED
    character_rev: Mapped[int] = mapped_column(Integer)
    persona_rev: Mapped[int] = mapped_column(Integer)
    memory_through_turn_seq: Mapped[int] = mapped_column(Integer, default=0)
    embedding_through_turn_seq: Mapped[int] = mapped_column(Integer, default=0)
    summary_through_turn_seq: Mapped[int] = mapped_column(Integer, default=0)
    last_turn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LifecycleCursor(Base):
    """Highest contiguous ``aggregateSeq`` applied per BE aggregate (gap = hold)."""

    __tablename__ = "lifecycle_cursor"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    aggregate_type: Mapped[str] = mapped_column(String(16), primary_key=True)
    aggregate_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    applied_seq: Mapped[int] = mapped_column(BigInteger, default=0)


class EventInbox(Base):
    __tablename__ = "event_inbox"
    __table_args__ = (
        Index("ix_inbox_pending", "status", "tenant_id", "aggregate_type", "aggregate_id", "aggregate_seq"),
        Index("ix_inbox_conversation", "tenant_id", "conversation_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)  # receiptId
    event_id: Mapped[str] = mapped_column(String(64), unique=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    owner_id: Mapped[str] = mapped_column(String(64))
    aggregate_type: Mapped[str] = mapped_column(String(16))
    aggregate_id: Mapped[str] = mapped_column(String(64))
    aggregate_seq: Mapped[int] = mapped_column(BigInteger)
    type: Mapped[str] = mapped_column(String(48))
    conversation_id: Mapped[str | None] = mapped_column(String(64))  # scrub lookup on reset/delete
    payload: Mapped[dict[str, Any]]
    status: Mapped[str] = mapped_column(String(16), default="RECEIVED")
    error: Mapped[str | None] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SettingsSnapshot(ScopedMixin, TimestampMixin, Base):
    """Immutable copy of character card + persona per revision pair."""

    __tablename__ = "settings_snapshot"
    __table_args__ = (
        UniqueConstraint("tenant_id", "conversation_id", "epoch", "character_rev", "persona_rev"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    character_rev: Mapped[int] = mapped_column(Integer)
    persona_rev: Mapped[int] = mapped_column(Integer)
    character_card: Mapped[dict[str, Any]]
    persona: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    opening_scene: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class CommittedTurn(ScopedMixin, TimestampMixin, Base):
    """Projection of BE-confirmed turns. Turn 0 is the opening scene."""

    __tablename__ = "committed_turn"
    __table_args__ = (UniqueConstraint("tenant_id", "conversation_id", "epoch", "turn_seq"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    turn_seq: Mapped[int] = mapped_column(Integer)
    user_msg_id: Mapped[str | None] = mapped_column(String(64))
    user_text: Mapped[str | None] = mapped_column(Text)
    assistant_msg_id: Mapped[str | None] = mapped_column(String(64))
    assistant_text: Mapped[str] = mapped_column(Text)
    character_rev: Mapped[int] = mapped_column(Integer)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Revocation(TimestampMixin, Base):
    """Minimal tombstone that survives deletion, so late events / restores cannot resurrect data."""

    __tablename__ = "revocation"
    __table_args__ = (Index("ix_revocation_lookup", "tenant_id", "scope", "target_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64))
    scope: Mapped[str] = mapped_column(String(16))  # CHARACTER | CONVERSATION | EPOCH
    target_id: Mapped[str] = mapped_column(String(64))
    epoch: Mapped[int | None] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(48))


# ---------------------------------------------------------------- derived state


class Memory(ScopedMixin, TimestampMixin, Base):
    __tablename__ = "memory"
    __table_args__ = (Index("ix_memory_scope_valid", "tenant_id", "conversation_id", "epoch", "valid_to"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    type: Mapped[str] = mapped_column(String(24))
    subject: Mapped[str] = mapped_column(String(32))
    predicate: Mapped[str] = mapped_column(String(64))
    value: Mapped[str] = mapped_column(Text)
    time_expr: Mapped[str | None] = mapped_column(String(64))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    time_uncertainty: Mapped[str | None] = mapped_column(String(16))
    is_fiction: Mapped[bool] = mapped_column(Boolean, default=False)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    source_msg_ids: Mapped[list[Any]]
    source_turn_seq: Mapped[int] = mapped_column(Integer)
    evidence: Mapped[str] = mapped_column(Text)
    supersedes_id: Mapped[str | None] = mapped_column(String(36))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    conflicts_with_revision: Mapped[int | None] = mapped_column(Integer)
    extractor_version: Mapped[str] = mapped_column(String(32))
    embedding = mapped_column(Vector(), nullable=True)
    embedding_version: Mapped[str | None] = mapped_column(String(64))
    index_status: Mapped[str] = mapped_column(String(16), default="PENDING")  # PENDING | INDEXED | FAILED


class UnfinishedEvent(ScopedMixin, TimestampMixin, Base):
    __tablename__ = "unfinished_event"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    memory_id: Mapped[str] = mapped_column(String(36))
    status: Mapped[str] = mapped_column(String(16), default="OPEN")  # OPEN | RESOLVED | CANCELLED
    title: Mapped[str] = mapped_column(Text)
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_turn_seq: Mapped[int | None] = mapped_column(Integer)


class RelationshipEvent(ScopedMixin, TimestampMixin, Base):
    __tablename__ = "relationship_event"
    __table_args__ = (UniqueConstraint("tenant_id", "conversation_id", "epoch", "event_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    event_key: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(32))
    delta: Mapped[dict[str, Any]]
    source_turn_seq: Mapped[int] = mapped_column(Integer)
    source_msg_ids: Mapped[list[Any]]
    evidence: Mapped[str] = mapped_column(Text)
    reducer_version: Mapped[str] = mapped_column(String(32))


class RelationshipState(ScopedMixin, TimestampMixin, Base):
    __tablename__ = "relationship_state"
    __table_args__ = (UniqueConstraint("tenant_id", "conversation_id", "epoch"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    trust: Mapped[float] = mapped_column(Float)
    closeness: Mapped[float] = mapped_column(Float)
    tension: Mapped[float] = mapped_column(Float)
    stage: Mapped[str] = mapped_column(String(16))
    baseline_source: Mapped[str] = mapped_column(String(16), default="DEFAULT")  # DEFAULT | MODEL
    daily_gain: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    through_turn_seq: Mapped[int] = mapped_column(Integer, default=0)


class Summary(ScopedMixin, TimestampMixin, Base):
    __tablename__ = "summary"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    from_turn_seq: Mapped[int] = mapped_column(Integer)
    to_turn_seq: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    version: Mapped[str] = mapped_column(String(32))


class ContextSnapshot(ScopedMixin, TimestampMixin, Base):
    __tablename__ = "context_snapshot"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    turn_seq: Mapped[int] = mapped_column(Integer)
    content: Mapped[dict[str, Any]]
    input_hash: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(32))
    memory_ids: Mapped[list[Any]]
    degraded: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


# ---------------------------------------------------------------- execution


class AiJob(TimestampMixin, Base):
    """Generation / memory / baseline jobs. Claimed with FOR UPDATE SKIP LOCKED + lease + fencing token."""

    __tablename__ = "ai_job"
    __table_args__ = (
        # One active generation per conversation (plan §09 "연속 전송").
        Index(
            "uq_active_generation",
            "tenant_id",
            "conversation_id",
            unique=True,
            postgresql_where=text(
                "kind = 'GENERATION' AND status IN ('ACCEPTED','WAITING_CONTEXT','RUNNING','STREAMING')"
            ),
        ),
        Index("ix_job_claim", "kind", "status", "lease_until"),
        UniqueConstraint("kind", "attempt_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(16))  # GENERATION | MEMORY | BASELINE
    tenant_id: Mapped[str] = mapped_column(String(64))
    owner_id: Mapped[str] = mapped_column(String(64))
    conversation_id: Mapped[str] = mapped_column(String(64))
    epoch: Mapped[int] = mapped_column(Integer)
    turn_seq: Mapped[int] = mapped_column(Integer)
    attempt_id: Mapped[str] = mapped_column(String(64))
    response_id: Mapped[str | None] = mapped_column(String(64))
    payload_hash: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]]
    mode: Mapped[str | None] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="ACCEPTED")
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(64))
    fencing_token: Mapped[int] = mapped_column(BigInteger, default=0)
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    snapshot_id: Mapped[str | None] = mapped_column(String(36))
    next_event_seq: Mapped[int] = mapped_column(Integer, default=1)
    result_text: Mapped[str | None] = mapped_column(Text)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    degraded: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    error_code: Mapped[str | None] = mapped_column(String(32))
    error_detail: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class StreamEvent(Base):
    __tablename__ = "stream_event"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    event: Mapped[str] = mapped_column(String(16))
    data: Mapped[dict[str, Any]]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UsageLedger(TimestampMixin, Base):
    __tablename__ = "usage_ledger"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64))
    owner_id: Mapped[str] = mapped_column(String(64))
    conversation_id: Mapped[str] = mapped_column(String(64))
    attempt_id: Mapped[str] = mapped_column(String(64))
    purpose: Mapped[str] = mapped_column(String(16))  # gen | extract | summary | embed | baseline
    model: Mapped[str] = mapped_column(String(64))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    est_cost: Mapped[float] = mapped_column(Float, default=0.0)
