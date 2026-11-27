# 평가셋 (plan §14)

- `scenarios/*.json` : 시나리오 1개 = 파일 1개. 지금은 §30 골든 시나리오 1개만 있습니다.
- `expect.context` 는 결정적 검사(Context에 무엇이 들어갔는지)이고, `response_rubric` 은 사람/보조 모델 평가 항목입니다.
- 골든 시나리오의 Context 검사는 `tests/integration/test_lifecycle_memory.py::test_golden_scenario_interview_recalled_on_day7` 로 매 빌드 실행됩니다.

다음 단계(P0~P1): 시나리오 20개 초안(캐릭터 일관성, 기억, 정정, 설정 변경 시도, 초기화/삭제, 인젝션) 추가 → 실제 모델 2~3종 비교 러너(W2) 작성.
