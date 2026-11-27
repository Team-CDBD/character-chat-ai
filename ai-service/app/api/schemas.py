"""BE-AI contract v1 (plan §10).

These Pydantic models are the single source of truth for ``contracts/``: running
``python -m tools.export_contracts`` regenerates the OpenAPI document and JSON Schemas
that are shared with the BE team.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

SCHEMA_VERSION = "1.0"


class ContractModel(BaseModel):
    """camelCase on the wire, snake_case in Python."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


# ---------------------------------------------------------------- settings payloads


class CharacterCard(ContractModel):
    name: str = Field(max_length=50)
    intro: str = Field(default="", max_length=300)
    personality: str = Field(default="", max_length=1000)
    speech_style: str = Field(default="", max_length=1000)
    background: str = Field(default="", max_length=1500)
    relationship_to_user: str = Field(default="", max_length=1000)
    extra: str = Field(default="", max_length=1500)


class UserPersona(ContractModel):
    name: str = Field(max_length=50)
    age: str | None = Field(default=None, max_length=20)
    job: str | None = Field(default=None, max_length=50)
    description: str = Field(default="", max_length=1000)


class OpeningScene(ContractModel):
    message_id: str
    narration: str = Field(default="", max_length=2000)
    character_line: str = Field(default="", max_length=1000)


class ChatMessage(ContractModel):
    id: str
    text: str = Field(max_length=4000)
    sent_at: datetime


class AssistantMessage(ContractModel):
    id: str
    response_id: str
    text: str = Field(max_length=8000)


# ---------------------------------------------------------------- generation


class GenerationMode(StrEnum):
    NEW = "NEW"
    REGENERATE = "REGENERATE"
    RETRY = "RETRY"


class JobStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    WAITING_CONTEXT = "WAITING_CONTEXT"
    RUNNING = "RUNNING"
    STREAMING = "STREAMING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REVOKED = "REVOKED"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.REVOKED)


ACTIVE_STATUSES = tuple(s.value for s in JobStatus if not s.terminal)


class ErrorCode(StrEnum):
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    CONTEXT_NOT_READY = "CONTEXT_NOT_READY"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    SNAPSHOT_EXPIRED = "SNAPSHOT_EXPIRED"
    REVOKED = "REVOKED"
    INTERNAL = "INTERNAL"

    @property
    def retryable(self) -> bool:
        """Hint for BE's Retry decision (plan §09). BE stays the final decider."""
        return self in (
            ErrorCode.PROVIDER_TIMEOUT,
            ErrorCode.PROVIDER_ERROR,
            ErrorCode.CONTEXT_NOT_READY,
            ErrorCode.INTERNAL,
        )


class GenerationRequest(ContractModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    tenant_id: str
    owner_id: str
    conversation_id: str
    epoch: int = Field(ge=1)
    turn_seq: int = Field(ge=1)
    attempt_id: str
    response_id: str
    mode: GenerationMode = GenerationMode.NEW
    required_lifecycle_seq: int = Field(ge=0)
    base_committed_turn_seq: int = Field(ge=0)
    character_revision: int = Field(ge=1)
    persona_revision: int = Field(ge=0)
    user_message: ChatMessage
    context_snapshot_id: str | None = None
    timezone: str = "Asia/Seoul"
    generation_policy_version: str = "chat-v1"
    deadline_at: datetime
    trace_id: str | None = None


class GenerationAccepted(ContractModel):
    ai_job_id: str
    status: JobStatus


class Usage(ContractModel):
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None


class ErrorBody(ContractModel):
    code: ErrorCode
    message: str
    retryable: bool


class GenerationResult(ContractModel):
    ai_job_id: str
    attempt_id: str
    response_id: str
    status: JobStatus
    text: str | None = None
    usage: Usage | None = None
    error: ErrorBody | None = None
    context_snapshot_id: str | None = None
    degraded: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- lifecycle events


class AggregateType(StrEnum):
    CONVERSATION = "CONVERSATION"
    CHARACTER = "CHARACTER"


class _EventBase(ContractModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    event_id: str
    tenant_id: str
    owner_id: str
    aggregate_type: AggregateType
    aggregate_id: str
    aggregate_seq: int = Field(ge=1)
    occurred_at: datetime


class ConversationOpenedPayload(ContractModel):
    conversation_id: str
    character_id: str
    epoch: int = Field(ge=1)
    character_revision: int = Field(ge=1)
    character: CharacterCard
    persona_revision: int = Field(ge=0)
    persona: UserPersona | None = None
    opening_scene: OpeningScene | None = None


class AnswerCommittedPayload(ContractModel):
    conversation_id: str
    epoch: int = Field(ge=1)
    turn_seq: int = Field(ge=1)
    character_revision: int = Field(ge=1)
    user_message: ChatMessage
    assistant_message: AssistantMessage


class CharacterRevisionChangedPayload(ContractModel):
    conversation_id: str
    epoch: int = Field(ge=1)
    character_revision: int = Field(ge=1)
    character: CharacterCard
    persona_revision: int = Field(ge=0)
    persona: UserPersona | None = None


class ConversationResetPayload(ContractModel):
    conversation_id: str
    old_epoch: int = Field(ge=1)
    new_epoch: int = Field(ge=2)


class CharacterDeletedPayload(ContractModel):
    character_id: str
    conversation_ids: list[str] = Field(default_factory=list)


class ConversationOpened(_EventBase):
    type: Literal["ConversationOpened"]
    payload: ConversationOpenedPayload


class AnswerCommitted(_EventBase):
    type: Literal["AnswerCommitted"]
    payload: AnswerCommittedPayload


class CharacterRevisionChanged(_EventBase):
    type: Literal["CharacterRevisionChanged"]
    payload: CharacterRevisionChangedPayload


class ConversationReset(_EventBase):
    type: Literal["ConversationReset"]
    payload: ConversationResetPayload


class CharacterDeleted(_EventBase):
    type: Literal["CharacterDeleted"]
    payload: CharacterDeletedPayload


LifecycleEvent = Annotated[
    ConversationOpened | AnswerCommitted | CharacterRevisionChanged | ConversationReset | CharacterDeleted,
    Field(discriminator="type"),
]


class LifecycleReceiptStatus(StrEnum):
    RECEIVED = "RECEIVED"
    APPLIED = "APPLIED"
    FAILED = "FAILED"


class LifecycleAccepted(ContractModel):
    receipt_id: str


class LifecycleReceipt(ContractModel):
    receipt_id: str
    event_id: str
    status: LifecycleReceiptStatus
    error: str | None = None
    applied_at: datetime | None = None


class ConversationState(ContractModel):
    conversation_id: str
    epoch: int
    status: str
    applied_lifecycle_seq: int
    memory_through_turn_seq: int
    embedding_through_turn_seq: int
    character_revision: int
    persona_revision: int


# ---------------------------------------------------------------- SSE


class SseEventName(StrEnum):
    ACCEPTED = "accepted"
    DELTA = "delta"
    COMPLETED = "completed"
    FAILED = "failed"


def error_body(code: ErrorCode, message: str) -> dict[str, Any]:
    return ErrorBody(code=code, message=message, retryable=code.retryable).model_dump(by_alias=True)
