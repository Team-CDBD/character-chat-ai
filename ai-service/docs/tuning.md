# 튜닝 값 (데이터로 결정할 값)

아래 값은 실제 대화 데이터나 평가셋으로 정해야 하는 값입니다. 아직 데이터가 없어서 **지금 동작하는 값을 기본값으로 고정**해 두었고, 코드 변경 없이 바꿀 수 있게 했습니다. 값의 원본은 `app/tuning.py` 입니다.

## 바꾸는 방법

| 방법 | 예시 |
|---|---|
| JSON 파일 (바꿀 키만) | `AI_TUNING_FILE=/srv/tuning.json` |
| 값 하나씩 환경변수 | `AI_TUNING__RETRIEVAL__W_SEMANTIC=0.6` |
| 블록 전체 JSON 환경변수 | `AI_TUNING='{"relationship": {"daily_positive_cap": 8}}'` |

환경변수가 파일보다 우선합니다. `deltas`, `budgets` 처럼 사전인 값은 바꾼 키만 덮어쓰고 나머지는 기본값을 유지합니다. 모르는 키는 시작할 때 오류가 납니다(오타 방지).

```sh
python -m tools.show_tuning      # 지금 적용된 값과 버전 해시 확인
```

적용된 값의 해시(`tuning-xxxxxxxxxx`)는 Context 스냅샷의 `content.tuningVersion`에 함께 남습니다. 평가를 돌릴 때 어떤 값으로 만든 답변인지 구분하는 데 씁니다. API와 워커는 같은 값을 써야 하므로 둘 다 같은 설정으로 재시작합니다.

파일 예시:

```json
{
  "retrieval": {"w_semantic": 0.6, "w_keyword": 0.1},
  "relationship": {"deltas": {"SUPPORT": [0, 1, 0]}, "daily_positive_cap": 8},
  "context": {"budgets": {"recent": 3200}}
}
```

## 값 목록

### retrieval — 기억 검색 점수 (`knowledge/retriever.py`)
결정 근거: 기억 평가셋의 recall@k / precision, "엉뚱한 회상" 비율.

| 키 | 기본값 | 의미 |
|---|---|---|
| `w_semantic` | 0.55 | 임베딩 유사도 가중치 |
| `w_keyword` | 0.15 | 키워드 겹침 가중치 |
| `w_importance` | 0.15 | 추출 시 매긴 중요도 가중치 |
| `w_time` | 0.15 | 시간 근접도 가중치 |
| `recency_decay_days` | 30 | 일반 기억의 시간 점수 `exp(-경과일/N)` |
| `schedule_full_days` | 2 | 일정은 해석 일시 ±N일이면 시간 점수 1.0 |
| `schedule_fade_days` | 12 | 그 뒤 N일에 걸쳐 0으로 감소 |
| `candidate_limit` | 500 | 질의마다 점수를 매기는 최근 기억 수 |

검색 상위 개수와 게이트는 기존대로 `AI_RETRIEVAL_TOP_K`(8), `AI_RETRIEVAL_GATE`(0.25), `AI_UNFINISHED_GATE`(0.30)로 바꿉니다.

### unfinished — 약속·일정 노출 (`relationship/unfinished.py`)
결정 근거: 후속 질문이 자연스러웠는지에 대한 평가, 노출 대비 사용자 반응.

| 키 | 기본값 | 의미 |
|---|---|---|
| `due_window_hours` | 24 | 예정 시각 ±N시간이면 DUE (우선 노출) |
| `expire_after_days` | 14 | 날짜 있는 사건이 예정일 뒤 N일 지나면 EXPIRED |
| `undated_expire_after_days` | 28 | 날짜 없는 사건은 생성 뒤 N일 |
| `max_exposed` | 2 | 한 턴에 Context로 넣는 최대 개수 |

### relationship — 관계 Reducer (`relationship/reducer.py`)
결정 근거: 긴 시나리오(100턴)에서 단계가 바뀌는 속도, 말투 변화에 대한 평가.

| 키 | 기본값 | 의미 |
|---|---|---|
| `deltas` | 아래 표 | 사건별 (신뢰, 친밀, 긴장) 변화량 |
| `daily_positive_cap` | 6 | SHARED_EXPERIENCE·SUPPORT로 하루에 오를 수 있는 친밀도 |
| `hysteresis` | 3 | 단계 경계 왕복 방지 폭 |
| `tense_tension` | 50 | 긴장 ≥ N이면 TENSE |
| `close_closeness` / `close_trust` | 60 / 55 | 둘 다 넘으면 CLOSE |
| `familiar_closeness` | 35 | 친밀 ≥ N이면 FAMILIAR |
| `default_trust` / `default_closeness` / `default_tension` | 50 / 40 / 0 | BASELINE 모델 호출 전·실패 시 초기값 |

| 사건 | 신뢰 | 친밀 | 긴장 |
|---|---|---|---|
| PERSONAL_DISCLOSURE | +4 | +2 | 0 |
| PROMISE_KEPT | +5 | 0 | 0 |
| PROMISE_BROKEN | -4 | 0 | +3 |
| CONFLICT | 0 | 0 | +6 |
| RECONCILIATION | 0 | +3 | -8 |
| SHARED_EXPERIENCE | 0 | +3 | 0 |
| SUPPORT | 0 | +2 | 0 |

### context — Context 예산 (`knowledge/context_builder.py`)
결정 근거: 스냅샷의 `layerTokens` 실측 분포, 잘림 지점의 답변 품질, 모델 문맥 길이.

| 키 | 기본값 | 의미 |
|---|---|---|
| `budgets.policy` … `budgets.recent` | 400 / 1200 / 400 / 500 / 900 / 800 / 60 / 3000 | 계층별 토큰 예산 (policy, character, persona, relationship, memories, summary, time, recent). 합계 7,260 |
| `output_tokens` | 600 | 답변 최대 토큰 |
| `chat_temperature` | 0.8 | 생성 온도 (Regenerate는 `AI_REGENERATE_TEMPERATURE_BUMP` 0.1 추가) |

Ollama에서는 `입력 예산 합계 + output_tokens`가 `AI_OLLAMA_NUM_CTX`(기본 8192)를 넘으면 시작할 때 오류를 냅니다.

### memory — 기억 파이프라인 (`memory/*.py`)
결정 근거: 추출 precision, 턴당 비용, 장애 시 재시도 비용.

| 키 | 기본값 | 의미 |
|---|---|---|
| `max_new_per_turn` | 5 | 한 턴에서 저장하는 최대 기억 수 (중요도 순) |
| `prefilter_min_chars` | 12 | 신호 단어가 없을 때 추출을 시도하는 최소 글자 수 |
| `max_runs` | 3 | 모델 오류 시 기억·관계·설정 검사 Job의 최대 시도 횟수 |
| `reindex_batch` | 16 | 기억 Job마다 같이 다시 색인하는 이전 기억 수 |
| `revision_check_batch` | 40 | 캐릭터 수정 후 한 번의 모델 호출로 검사하는 기억 수 |

## 운영 설정 (튜닝 값은 아니지만 함께 바꾸는 것)

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `AI_OLLAMA_NUM_CTX` | 8192 | Ollama 문맥 길이 |
| `AI_TOKENIZER_FILE` | 없음 | 채팅 모델의 Hugging Face `tokenizer.json`. 있으면 토큰 수를 정확히 셉니다 |
| `AI_TOKEN_CALIBRATION` | true | 토크나이저가 없을 때 공급자가 알려준 실제 토큰 수로 추정 비율을 보정 |
| `AI_REINDEX_INTERVAL_SECONDS` | 60 | 워커의 재색인 스윕 주기. 0이면 끔 |
| `AI_REINDEX_SWEEP_BATCH` | 32 | 스윕 한 번의 임베딩 호출당 기억 수 |
