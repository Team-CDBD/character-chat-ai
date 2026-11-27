# BE–AI 계약 v1 (`/internal/ai/v1`)

이 디렉터리는 `app/api/schemas.py` 에서 자동 생성됩니다. 직접 고치지 말고 `python -m tools.export_contracts` 로 다시 만드세요. CI는 `--check` 로 최신 여부를 검사합니다.

| 파일 | 내용 |
|---|---|
| `openapi.json` | 전체 엔드포인트 |
| `schemas/*.schema.json` | 요청·응답·Lifecycle 이벤트 JSON Schema |
| `fixtures/*.json` | 예시 payload (Mock BE·테스트와 동일한 빌더로 생성) |

## 인증
모든 요청에 `Authorization: Bearer <service token>`.

## SSE `GET /generations/{aiJobId}/events`
| event | data |
|---|---|
| `accepted` | `{aiJobId, attemptId}` |
| `delta` | `{text}` — Output Policy 검사를 통과한 문장 단위 청크 |
| `completed` | `{responseId, text, usage, contextSnapshotId}` |
| `failed` | `{error: {code, message, retryable}}` |

`id` 는 job 안에서 단조 증가하는 `eventSeq` 이며, 재접속 시 `Last-Event-ID` (또는 `?after=`) 이후부터 다시 보냅니다. 구독 종료는 Job 취소가 아닙니다.

## 오류 코드
`PROVIDER_TIMEOUT`, `PROVIDER_ERROR`, `CONTEXT_NOT_READY`, `INTERNAL` → `retryable=true` (BE가 새 attempt로 Retry 가능)
`POLICY_BLOCKED`, `BUDGET_EXCEEDED`, `SNAPSHOT_EXPIRED`, `REVOKED` → `retryable=false`

HTTP 오류: `401 UNAUTHORIZED`, `404 CONVERSATION_NOT_REGISTERED`, `409 IDEMPOTENCY_CONFLICT | ACTIVE_GENERATION_EXISTS | EPOCH_REVOKED`

## Lifecycle 이벤트 순서
`(aggregateType, aggregateId)` 별 `aggregateSeq` 가 1부터 빈틈없이 증가해야 합니다. 빈 번호가 있으면 뒤 이벤트는 `RECEIVED` 로 보류됩니다. 대화 이벤트는 `CONVERSATION/{conversationId}`, 캐릭터 삭제는 `CHARACTER/{characterId}` 입니다.
