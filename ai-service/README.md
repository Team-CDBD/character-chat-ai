# ai-service

캐릭터 챗 MVP의 **독립 AI 서비스**입니다. 설계 기준은 프로젝트 문서 「AI 아키텍처와 AI 서버 개발 플랜 v1.0」(이하 plan)이며, BE와는 버전이 있는 계약 `/internal/ai/v1` 으로만 통신합니다.

- Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2 (async) · PostgreSQL 16 + pgvector · Alembic
- 작업 실행: DB Job 테이블 + 자체 Worker 루프 (`FOR UPDATE SKIP LOCKED`, lease, fencing token). Redis·브로커 없음
- 모델: 아직 미정 → `MockProvider` 로 전 경로가 동작. 실제 공급자는 W2 한국어 평가 후 `app/providers/` 에 추가

## 빠르게 실행하기

```bash
# 1) docker로 전체 실행 (api + worker + postgres/pgvector)
docker compose -f infra/docker-compose.yml up --build

# 2) 다른 터미널에서 Mock BE로 대화 재생 (SSE 스트리밍 확인)
pip install -e '.[dev]'
python -m tools.mock_be demo --base-url http://localhost:8080 --token local-dev-token
```

로컬 Python으로 직접 띄울 때:

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -e '.[dev]'
export AI_DATABASE_URL=postgresql+asyncpg://ai:ai@localhost:5432/ai
.venv/bin/alembic upgrade head
.venv/bin/uvicorn app.main:app --port 8080      # ai-api
.venv/bin/python -m app.workers.runner          # ai-worker (생성 8 슬롯 + 기억 4 슬롯 + lifecycle 루프)
```

## 테스트

```bash
# pgvector가 있는 PostgreSQL 필요 (없으면 통합 테스트는 skip)
export AI_TEST_DATABASE_URL=postgresql+asyncpg://ai:ai@localhost:5432/ai_test
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python -m tools.export_contracts --check   # contracts/ 최신 여부
```

- `tests/unit` : 결정적 로직 (관계 Reducer, 기억 검증·Reducer, Context Builder 예산, Output Policy, 미완료 사건, 계약 fixture)
- `tests/integration` : 실제 DB + Mock Provider로 API·Worker 전체 흐름 (멱등 접수, 활성 Job 1개, SSE 재전송, Regenerate 스냅샷 재사용, fencing, 순서 보장, 골든 시나리오, 정정, False Memory 차단, 초기화·삭제 후 부활 방지)

## 구조 (plan §05 저장소 구조)

```
contracts/          OpenAPI · JSON Schema · fixtures (자동 생성, BE와 공유)
app/api/            FastAPI 라우터, 서비스 인증, 계약 스키마(schemas.py = 계약의 원본)
app/dialogue/       Dialogue Manager: 경계 검증 → Context 대기 → Snapshot → 생성 (자율 Agent 루프 없음)
app/understanding/  입력 정규화, 지시어("그때/거기") 검색 힌트 — 규칙 기반
app/knowledge/      Retriever(하이브리드 점수 + 게이트), Context Builder(계층 예산), Composer, Output Policy, Prompt Registry
app/memory/         Pre-filter, 추출 결과 근거 검증, Memory Reducer(supersedes·중복), 기억 Pipeline(+요약·임베딩)
app/relationship/   결정적 Relationship Reducer(변화량 표·상한·사건키·히스테리시스), 단계→행동 지침, 미완료 사건
app/lifecycle/      Inbox, aggregateSeq 순서 보장, 초기화·삭제(차단표식 → fencing → 정리)
app/providers/      Provider Adapter(timeout·동시성 슬롯·usage 표준화) + Mock
app/persistence/    SQLAlchemy 모델, Job claim/lease/fencing, Alembic 마이그레이션
app/workers/        lifecycle 루프 · 생성 슬롯 · 기억 슬롯 (한 프로세스, 슬롯 분리)
prompts/            버전 관리 프롬프트 (chat-v1, extract-v1, summary-v1, baseline-v1)
evals/              평가셋 (골든 시나리오부터)
tools/              Mock BE CLI, 계약 export
infra/              docker-compose (로컬)
```

## 핵심 규칙이 코드에서 지켜지는 곳

| plan 결정 | 위치 |
|---|---|
| ① 설정은 대화로 바뀌지 않는다 | `prompts/chat-v1.policy.txt`, `knowledge/context_builder.py` (설정 계층 1~3은 절대 자르지 않고, 대화는 설정 계층에 들어가지 않음) |
| ② 확정된 답변만 기억 | 기억 Job은 `AnswerCommitted` 수신 시에만 생성 (`lifecycle/handler.py`) |
| ③ 모델은 제안, 코드가 결정 | `memory/validator.py` (근거·화자·허구), `memory/reducer.py`, `relationship/reducer.py` |
| ④ Regenerate = 같은 Context | `context_snapshot` 영속화 후 재사용, 없으면 `SNAPSHOT_EXPIRED` (조용히 재생성 안 함) |
| ⑤ 자연스러울 때만 회상 | `knowledge/retriever.py` 관련성 게이트, `relationship/unfinished.py` (DUE 우선, 최대 2개, 최근 언급 제외) |
| ⑥ 관계는 말투로 | `relationship/reducer.py::guideline` — 프롬프트에 수치 없음 |
| ⑦ 최근 대화 창으로 기억 지연 흡수 | `dialogue/manager.py::wait_context` — lifecycle seq는 대기, 기억 watermark는 최대 3초 후 창 확장 + degraded |

## 단계별 진행 상황 (plan §15)

| 단계 | 상태 |
|---|---|
| P0 계약·Mock·평가셋 | ✅ 계약 v1 스키마·fixture, Mock BE CLI, Mock Provider, docker-compose, 골든 시나리오 1개 (평가셋 20개는 남음) |
| P1 기본 생성 경로 | ✅ 접수·조회·SSE·서비스 인증, Job lease/fencing, StreamEvent 재전송, Inbox·Projection / ⬜ 실제 모델 연결, 모델 비교 리포트 |
| P2 캐릭터 품질 | 🟡 chat-v1, Context Builder v1, Snapshot·Regenerate·Retry 분류, 청크 검사·역할 이탈 지표, revision 반영 / ⬜ 충돌 기억 표식(`conflicts_with_revision`) |
| P3 장시간 대화 | 🟡 최근 창·축소 순서·Rolling Summary 기본 구현 / ⬜ 100턴 시나리오 검증 |
| P4 Long-term Memory | 🟡 Pre-filter·근거 검증·supersedes·하이브리드 검색·게이트 / ⬜ 실제 추출 모델·precision 리포트, 임베딩 별도 Job 분리 |
| P5 Relationship | 🟡 결정적 Reducer·단계·지침·미완료 사건 수명주기 / ⬜ 실제 모델 baseline, 튜닝 |
| P6 초기화·삭제 | 🟡 차단표식·fencing·정리·늦은 이벤트 무시 / ⬜ 삭제 영수증 상세, 지표 대시보드 |
| P7 통합·부하 | ⬜ |

## BE와 합의가 필요한 가정 (이번 코드에서 임시로 정한 것)

1. Lifecycle 이벤트 집계 단위: 대화 이벤트는 `CONVERSATION/{conversationId}`, 캐릭터 삭제는 `CHARACTER/{characterId}` 로 각자 `aggregateSeq` 를 1부터 셉니다.
2. `CharacterRevisionChanged` 는 대화 단위로 발행하고 캐릭터 카드·Persona 전체 사본을 담습니다 (Persona 변경도 같은 이벤트).
3. 첫 장면은 `ConversationOpened.openingScene` 으로 받아 AI가 Turn 0으로 기록합니다. 초기화 시 BE는 `ConversationReset(oldEpoch, newEpoch)` 만 보내면 AI가 기존 설정·첫 장면으로 새 epoch를 엽니다.
4. 오류 코드와 `retryable` 힌트는 `contracts/README.md` 표를 따릅니다. 최종 Retry 판단은 BE.

## OpenAI로 실행

기본 공급자는 mock이다. OpenAI를 사용할 때는 아래 환경변수와
`AI_OPENAI_API_KEY`를 실행 환경의 비밀 설정에 지정한다.
채팅·추출 모델은 Chat Completions 스트리밍, temperature 및 JSON 객체 출력을 지원해야 한다.

```sh
export AI_PROVIDER=openai
export AI_CHAT_MODEL=gpt-4o-mini
export AI_EXTRACT_MODEL=gpt-4o-mini
export AI_EMBEDDING_MODEL=text-embedding-3-small
docker compose -f infra/docker-compose.yml up --build
```

실제 대화와 기억 추출 입력은 설정한 제공자에게 전송되고 API 요금이 발생한다.
생성 사용량은 공급자가 보고한 값을 우선하며 누락 시 기존 토큰 추정치를 사용한다.
