# character-chat-ai

캐릭터 챗 MVP의 AI 서비스 저장소입니다.

- [`ai-service/`](ai-service/README.md) — 독립 AI 서비스 (FastAPI + PostgreSQL/pgvector). BE와는 `/internal/ai/v1` 계약으로만 통신합니다.
- [`ai-service/contracts/`](ai-service/contracts/README.md) — BE와 공유하는 OpenAPI·JSON Schema·fixture.
