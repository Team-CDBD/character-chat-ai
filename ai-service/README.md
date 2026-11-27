# ai-service

캐릭터 챗 MVP의 **독립 AI 서비스**입니다. 설계 기준은 프로젝트 문서 「AI 아키텍처와 AI 서버 개발 플랜 v1.0」(이하 plan)이며, BE와는 버전이 있는 계약 `/internal/ai/v1` 으로만 통신합니다.

- Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2 (async) · PostgreSQL 16 + pgvector · Alembic
- 작업 실행: DB Job 테이블 + 자체 Worker 루프 (`FOR UPDATE SKIP LOCKED`, lease, fencing token). Redis·브로커 없음
- 모델: 기본 Ollama (Gemma 4 12B), 선택 OpenAI 또는 Mock

## 빠르게 실행하기

```bash
# 1) Ollama 기동 및 최초 모델 다운로드
docker compose -f infra/docker-compose.yml --profile ollama up -d ollama
docker compose -f infra/docker-compose.yml exec ollama ollama pull gemma4:12b
docker compose -f infra/docker-compose.yml exec ollama ollama pull embeddinggemma

# 2) API, worker, PostgreSQL 기동
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
.venv/bin/python -m app.workers.runner          # ai-worker (생성·기억 슬롯과 lifecycle 루프)
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
| P1 기본 생성 경로 | ✅ 접수·조회·SSE·서비스 인증, Job lease/fencing, StreamEvent 재전송, Inbox·Projection / ⬜ 실기기 추론 검증, 모델 비교 리포트 |
| P2 캐릭터 품질 | 🟡 chat-v1, Context Builder v1, Snapshot·Regenerate·Retry 분류, 청크 검사·역할 이탈 지표, revision 반영, 충돌 기억만 숨기는 설정 검사(`REVISION` Job) / ⬜ 일관성 평가 기준선 |
| P3 장시간 대화 | 🟡 최근 창·축소 순서·Rolling Summary 기본 구현 / ⬜ 100턴 시나리오 검증 |
| P4 Long-term Memory | 🟡 Pre-filter·근거 검증·supersedes·하이브리드 검색·게이트, 임베딩 모델 변경 재색인 / ⬜ 실제 추출 모델·precision 리포트, 임베딩 별도 Job 분리 |
| P5 Relationship | 🟡 결정적 Reducer·단계·지침·미완료 사건 수명주기 / ⬜ 실제 모델 baseline, 튜닝 |
| P6 초기화·삭제 | 🟡 차단표식·fencing·정리·늦은 이벤트 무시 / ⬜ 삭제 영수증 상세, 지표 대시보드 |
| P7 통합·부하 | ⬜ |

## BE와 합의가 필요한 가정 (이번 코드에서 임시로 정한 것)

1. Lifecycle 이벤트 집계 단위: 대화 이벤트는 `CONVERSATION/{conversationId}`, 캐릭터 삭제는 `CHARACTER/{characterId}` 로 각자 `aggregateSeq` 를 1부터 셉니다.
2. `CharacterRevisionChanged` 는 대화 단위로 발행하고 캐릭터 카드·Persona 전체 사본을 담습니다 (Persona 변경도 같은 이벤트).
3. 첫 장면은 `ConversationOpened.openingScene` 으로 받아 AI가 Turn 0으로 기록합니다. 초기화 시 BE는 `ConversationReset(oldEpoch, newEpoch)` 만 보내면 AI가 기존 설정·첫 장면으로 새 epoch를 엽니다.
4. 오류 코드와 `retryable` 힌트는 `contracts/README.md` 표를 따릅니다. 최종 Retry 판단은 BE.

## 공급자 전환

실행할 공급자는 `AI_PROVIDER=ollama|openai|mock`로 선택한다. 진행 중인 작업이 끝난 뒤 API와 워커를 같은 설정으로 재시작한다. Compose에서는 아래 명령으로 설정을 다시 적용한다.

```sh
AI_PROVIDER=mock docker compose -f infra/docker-compose.yml up -d --build api worker
```

| 공급자 | 채팅·추출 기본 모델 | 임베딩 기본 모델 | 연결 |
|---|---|---|---|
| ollama | gemma4:12b | embeddinggemma | 로컬 Python은 localhost:11434, Compose는 ollama:11434 |
| openai | gpt-4o-mini | text-embedding-3-small | AI_OPENAI_API_KEY 필요 |
| mock | mock-chat / mock-extract | mock-embed | 오프라인 |

Ollama 기본값은 워커 프로세스당 모델 호출 1개, 문맥 8192토큰이다. GPU로 구동 중인 별도 Ollama도 `AI_OLLAMA_BASE_URL`을 컨테이너에서 접근 가능한 주소로 지정해 사용할 수 있다. 기본 Compose Ollama 서비스는 CPU로 실행된다. 사용자 장비에서 속도와 메모리는 별도 측정해야 한다.

모델을 직접 지정하려면 기존 공통 변수 `AI_CHAT_MODEL`, `AI_EXTRACT_MODEL`, `AI_EMBEDDING_MODEL`을 사용한다. 공급자를 바꿀 때 이 변수를 해제하면 표의 기본값을 따른다. 저장된 기억은 유지되며, 다른 임베딩 모델로 만든 벡터는 재색인이 끝날 때까지 키워드 검색으로 처리한다(아래 「임베딩 모델 변경과 재색인」).

## OpenAI로 실행

OpenAI를 사용할 때는 아래 환경변수와
`AI_OPENAI_API_KEY`를 실행 환경의 비밀 설정에 지정한다.
채팅·추출 모델은 Chat Completions 스트리밍, temperature 및 JSON 객체 출력을 지원해야 한다.

```sh
export AI_PROVIDER=openai
docker compose -f infra/docker-compose.yml up --build
```

실제 대화와 기억 추출 입력은 설정한 제공자에게 전송되고 API 요금이 발생한다.
생성 사용량은 공급자가 보고한 값을 우선하며, 누락 시 아래 「토큰 수 계산」의 카운터로 센다.

## 캐릭터 수정과 기억

캐릭터 카드가 바뀌면(`CharacterRevisionChanged`) 이전 대화에서 저장된 캐릭터·공동 기억을 **새 카드와 비교해 충돌하는 것만** 숨긴다.

1. 이벤트 적용 시 `REVISION` Job을 만든다. 기억 Job과 같은 슬롯에서 실행된다.
2. Job은 아직 현재 revision으로 검사하지 않은 기억을 모아, 바뀐 필드(이전 값 → 새 값)와 새 카드 전체를 추출 모델에 보낸다(`prompts/revision-check-v1.txt`).
3. 모델이 충돌이라고 답한 기억은 `conflicts_with_revision`이 찍혀 검색에서 빠지고, 나머지는 `checked_character_rev`가 현재 revision으로 올라가 계속 쓰인다. 모델이 보내지 않은 id는 무시한다. 답 형식이 틀리면(`conflicts` 목록이 없거나 id가 없는 항목) "충돌 없음"으로 보지 않고 모델 오류로 재시도한다.
4. 검사가 끝나기 전이나 모델이 계속 실패하면(최대 3회) 미검사 **캐릭터 기억은 숨긴 채로 둔다**(이전 동작과 같은 보수적 처리). 실패한 검사는 워커가 5분 뒤 다시 대기열에 넣는다(`AI_REVISION_RETRY_AFTER_SECONDS`). 공동 기억과 사용자 기억, 미완료 사건(약속·일정)은 검사 중에도 그대로 쓰인다.
5. 수정 전에 말한 턴의 추출이 수정 뒤에 끝나면, 그 기억에 대해 검사 Job을 한 번 더 만든다.
6. 수정을 되돌리면 이전에 충돌로 숨긴 기억도 다시 검사해, 새 카드와 맞으면 다시 쓴다.
7. 검사 Job은 대화마다 한 번에 하나만 돈다. 검사 중에 카드가 또 바뀌면 판정 결과를 버리고 최신 카드로 다시 판정한다. 기록은 항상 앞으로만 진행하므로 늦게 끝난 옛 검사가 새 결과를 덮지 않는다.
8. 0004 마이그레이션은 기존 데이터 중 이전 revision 때문에 숨겨져 있던 캐릭터 기억이 있는 활성 대화마다 검사 Job을 하나씩 넣는다.

비용: 카드를 수정할 때마다 그 캐릭터의 활성 대화마다 추출 모델을 기억 40개당 1회 호출한다. 기억 Job과 같은 슬롯을 쓰므로 인기 캐릭터를 수정하면 기억 처리가 잠시 밀릴 수 있다.

## 토큰 수 계산

Context 예산은 아래 순서로 고른 카운터로 센다. 워커 시작 로그(`token counting: ...`)와 Context 스냅샷의 `content.tokenCounter`에서 어떤 방식인지 확인할 수 있다.

| 순서 | 방식 | 조건 |
|---|---|---|
| 1 | `exact:hf` | `AI_TOKENIZER_FILE`에 채팅 모델의 Hugging Face `tokenizer.json`을 지정 (Gemma는 Hugging Face에서 약관 동의 후 받는다) |
| 2 | `exact:tiktoken` | OpenAI 공급자 + tiktoken 인코딩 파일을 받을 수 있는 환경 |
| 3 | `calibrated` | 그 외(Ollama 기본). 실제 호출에서 공급자가 알려준 입력 토큰 수(`prompt_eval_count`)로 글자/토큰 비율을 보정 |
| 4 | `estimate` | 보정 표본이 생기기 전, 그리고 Mock. `글자 수 / 1.5` |

보정은 **토큰을 넉넉하게** 세도록 만들었다. Ollama는 KV 캐시로 재사용한 앞부분을 입력 토큰에서 빼고 보고하므로, 그런 표본을 그대로 믿으면 토큰을 적게 세어 `num_ctx`를 넘길 수 있다.

- 표본이 10개 모이기 전에는 고정 추정치(1.5글자/토큰)를 쓴다.
- 최근 50개 표본의 하위 20% 비율에 여유 10%를 더한다.
- 학습한 비율은 기본값(1.5글자/토큰)을 넘지 않는다. 즉 보정은 토큰을 **더 많이** 세는 쪽으로만 작동한다(한국어를 잘게 나누는 토크나이저 대응). Ollama가 문맥 한도 때문에 오래된 메시지를 잘라 낸 호출은 구분이 어려워서 이렇게 막았다.
- 문맥 한도에서 잘린 호출은 표본에서 뺀다.
- 보정 값은 프로세스 메모리에만 있어 재시작하면 다시 추정치부터 시작한다.

시작할 때 `입력 예산 + 출력 + 채팅 템플릿 여유(메시지당 6토큰)`가 `AI_OLLAMA_NUM_CTX`를 넘으면 오류를 낸다. 운영에서는 보정보다 토크나이저 파일을 권장한다.
정확한 수가 필요하면 `pip install -e '.[tokenizer]'` 후 `AI_TOKENIZER_FILE`을 지정한다(Docker 이미지에는 이미 포함). Compose에서는 파일을 마운트하고 경로를 넘긴다.

## 임베딩 모델 변경과 재색인

`AI_EMBEDDING_MODEL`을 바꾸면 기존 기억의 벡터는 다른 모델로 만든 것이 되어 검색에서 쓰이지 않고 키워드 점수로 대신한다(`KEYWORD_FALLBACK`). 아래 세 경로가 새 모델로 다시 임베딩한다.

- 기억 Job: 같은 대화의 오래된 벡터·실패한 벡터를 Job마다 최대 16개씩 처리한다. 새 기억과는 따로 호출하므로 오래된 기억 하나가 계속 실패해도 새 기억 색인은 막히지 않는다.
- 워커 스윕: 모든 활성 대화를 대상으로 60초마다 32개씩 처리한다(`AI_REINDEX_INTERVAL_SECONDS`, `AI_REINDEX_SWEEP_BATCH`, 0이면 끔). 워커가 여러 대여도 PostgreSQL advisory lock으로 한 번에 하나만 돈다. 사용자 생성 요청이 대기열에 있으면(아직 시작 전) 스윕을 멈추고 다음 주기에 이어 간다(로컬 Ollama는 모델 슬롯이 하나라서). advisory lock은 PostgreSQL 직접 연결이 필요하다(pgbouncer transaction 모드 불가).
- 수동 실행:

```sh
python -m tools.reindex_embeddings --status   # 남은 개수 (버전 / 상태별)
python -m tools.reindex_embeddings            # 남은 것 전부 처리
# Compose
docker compose -f infra/docker-compose.yml exec worker python -m tools.reindex_embeddings
```

삭제·초기화로 지워진 기억이나 이미 다른 경로가 처리한 기억은 건너뛴다. 묶음 호출이 실패하면 한 개씩 다시 시도하고, 그래도 실패한 기억은 `FAILED`로 표시하고 다음 재시도 시각(`index_retry_at`)을 1분, 2분, 4분 … 최대 하루로 늘려 잡는다(마이그레이션 0005). 재시도는 생성 시각이 아니라 이 시각 순서로 고르므로, 계속 실패하는 기억이 뒤의 기억을 막지 않는다. 연속 3개가 모두 실패하면 공급자 장애로 보고 다음 주기로 미룬다. 실패 기록은 그 행이 아직 낡은 상태일 때만 남기므로, 기억 Job과 스윕이 같은 행을 동시에 처리해도 먼저 성공한 결과를 늦은 실패가 덮지 않는다.

주의할 점:
- 임베딩 버전은 모델 이름(`AI_EMBEDDING_MODEL`)으로만 구분한다. 같은 이름으로 `ollama pull`해서 모델 내용이 바뀌면 재색인이 일어나지 않으므로, 모델을 바꿀 때는 `embeddinggemma:<버전>`처럼 태그를 바꾼다.
- 배포 중 워커마다 `AI_EMBEDDING_MODEL`이 다르면 서로의 벡터를 낡은 것으로 보고 번갈아 다시 만든다. 모델을 바꾸는 배포는 모든 워커를 같은 값으로 맞춘 뒤 진행한다.
- 차원이 다른 벡터가 한 컬럼에 섞여도 검색은 현재 모델 버전 벡터끼리만 거리를 계산하므로 오류가 나지 않는다. 나중에 ANN 인덱스를 만들려면 재색인을 끝낸 뒤 차원을 고정해야 한다.

## 튜닝 값

검색 가중치, 관계 변화량·단계 경계, 미완료 사건 노출 기간, Context 예산 등 데이터로 정해야 할 값은 `app/tuning.py`에 지금 값을 기본값으로 모았다. 값 목록, 무엇으로 결정해야 하는지, 바꾸는 방법은 [docs/tuning.md](docs/tuning.md)에 있다.

```sh
AI_TUNING__RETRIEVAL__W_SEMANTIC=0.6 python -m tools.show_tuning
```
