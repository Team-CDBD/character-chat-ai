"""Mock BE (plan P0): builds contract payloads and replays a conversation against a running AI service.

    python -m tools.mock_be demo --base-url http://localhost:8080 --token local-dev-token

The builders are also used by the tests, so the fixtures, the tests and the CLI all speak
the same JSON as the BE will.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

HYUNWOO = {
    "name": "강현우",
    "intro": "10년 된 소꿉친구. 무뚝뚝하지만 은근히 나를 챙겨준다.",
    "personality": "말수가 적고 무뚝뚝하다. 감정을 직접 표현하지 않는다. 친한 사람은 행동으로 챙긴다.",
    "speechStyle": "반말. 문장이 짧다. 감탄사·이모티콘을 거의 쓰지 않는다.",
    "background": "사용자와 같은 동네에서 자랐다. 현재 같은 회사에 근무한다.",
    "relationshipToUser": "10년 된 소꿉친구. 현우는 오래전부터 사용자를 좋아하고 있다.",
    "extra": "질투할 때 말수가 더 줄어든다. 좋아한다는 사실을 먼저 고백하지 않는다.",
}
YEJIN = {"name": "예진", "job": "백엔드 개발자", "description": "현우와 중학교 때부터 알고 지냈다. 최근 취업 준비 중."}
SCENE = {"messageId": "scene-0", "narration": "퇴근길, 비가 쏟아진다. 현우가 우산을 들고 서 있다.", "characterLine": "우산 없지. 들어와."}


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class MockConversation:
    tenant_id: str = "tenant-1"
    owner_id: str = "user-7"
    conversation_id: str = field(default_factory=lambda: f"conv-{uuid.uuid4().hex[:8]}")
    character_id: str = field(default_factory=lambda: f"char-{uuid.uuid4().hex[:8]}")
    epoch: int = 1
    character_rev: int = 1
    persona_rev: int = 1
    seq: itertools.count = field(default_factory=lambda: itertools.count(1))
    last_seq: int = 0
    committed_turn: int = 0

    def _envelope(self, type_: str, payload: dict[str, Any], *, aggregate: str = "CONVERSATION",
                  aggregate_id: str | None = None, seq: int | None = None) -> dict[str, Any]:
        if seq is None:
            seq = next(self.seq)
            self.last_seq = seq
        return {
            "schemaVersion": "1.0",
            "eventId": f"evt-{uuid.uuid4().hex}",
            "tenantId": self.tenant_id,
            "ownerId": self.owner_id,
            "aggregateType": aggregate,
            "aggregateId": aggregate_id or self.conversation_id,
            "aggregateSeq": seq,
            "occurredAt": iso(datetime.now(UTC)),
            "type": type_,
            "payload": payload,
        }

    def opened(self, character: dict[str, Any] | None = None, persona: dict[str, Any] | None = None,
               scene: dict[str, Any] | None = SCENE) -> dict[str, Any]:
        return self._envelope("ConversationOpened", {
            "conversationId": self.conversation_id, "characterId": self.character_id, "epoch": self.epoch,
            "characterRevision": self.character_rev, "character": character or HYUNWOO,
            "personaRevision": self.persona_rev, "persona": persona or YEJIN, "openingScene": scene,
        })

    def committed(self, turn_seq: int, user_text: str, assistant_text: str, sent_at: datetime) -> dict[str, Any]:
        self.committed_turn = max(self.committed_turn, turn_seq)
        return self._envelope("AnswerCommitted", {
            "conversationId": self.conversation_id, "epoch": self.epoch, "turnSeq": turn_seq,
            "characterRevision": self.character_rev,
            "userMessage": {"id": f"u{turn_seq}", "text": user_text, "sentAt": iso(sent_at)},
            "assistantMessage": {"id": f"a{turn_seq}", "responseId": f"response-{turn_seq}", "text": assistant_text},
        })

    def revision_changed(self, character: dict[str, Any]) -> dict[str, Any]:
        self.character_rev += 1
        return self._envelope("CharacterRevisionChanged", {
            "conversationId": self.conversation_id, "epoch": self.epoch, "characterRevision": self.character_rev,
            "character": character, "personaRevision": self.persona_rev, "persona": YEJIN,
        })

    def reset(self) -> dict[str, Any]:
        old = self.epoch
        self.epoch += 1
        self.committed_turn = 0
        return self._envelope("ConversationReset", {"conversationId": self.conversation_id, "oldEpoch": old,
                                                    "newEpoch": self.epoch})

    def character_deleted(self) -> dict[str, Any]:
        return self._envelope("CharacterDeleted", {"characterId": self.character_id,
                                                   "conversationIds": [self.conversation_id]},
                              aggregate="CHARACTER", aggregate_id=self.character_id, seq=1)

    def generation(self, turn_seq: int, text: str, sent_at: datetime, *, mode: str = "NEW", attempt: str = "a",
                   snapshot_id: str | None = None, deadline: timedelta = timedelta(seconds=60)) -> dict[str, Any]:
        return {
            "schemaVersion": "1.0", "tenantId": self.tenant_id, "ownerId": self.owner_id,
            "conversationId": self.conversation_id, "epoch": self.epoch, "turnSeq": turn_seq,
            "attemptId": f"{self.conversation_id}-e{self.epoch}-t{turn_seq}-{attempt}",
            "responseId": f"response-{turn_seq}-{attempt}", "mode": mode,
            "requiredLifecycleSeq": self.last_seq, "baseCommittedTurnSeq": self.committed_turn,
            "characterRevision": self.character_rev, "personaRevision": self.persona_rev,
            "userMessage": {"id": f"u{turn_seq}", "text": text, "sentAt": iso(sent_at)},
            "contextSnapshotId": snapshot_id, "timezone": "Asia/Seoul", "generationPolicyVersion": "chat-v1",
            "deadlineAt": iso(datetime.now(UTC) + deadline), "traceId": f"trace-{turn_seq}",
        }


# ---------------------------------------------------------------- CLI


async def demo(base_url: str, token: str) -> None:
    import httpx

    headers = {"Authorization": f"Bearer {token}"}
    conv = MockConversation()
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=60) as client:
        r = await client.post("/internal/ai/v1/lifecycle-events", json=conv.opened())
        print("ConversationOpened ->", r.status_code, r.json())
        now = datetime.now(UTC)
        lines = ["나 다음 주 금요일에 면접 있어.", "요즘 취업 때문에 좀 힘들어", "오늘 점심 뭐 먹지"]
        for i, text in enumerate(lines, start=1):
            req = conv.generation(i, text, now + timedelta(minutes=i))
            r = await client.post("/internal/ai/v1/generations", json=req)
            job_id = r.json()["aiJobId"]
            print(f"\n[turn {i}] user: {text}\n  -> 202 {job_id}")
            answer = ""
            async with client.stream("GET", f"/internal/ai/v1/generations/{job_id}/events") as stream:
                async for line in stream.aiter_lines():
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line.split(":", 1)[1])
                        if event == "delta":
                            answer += data["text"]
                            sys.stdout.write(data["text"])
                            sys.stdout.flush()
                        elif event in ("completed", "failed"):
                            print(f"\n  <- {event}")
                            break
            r = await client.post("/internal/ai/v1/lifecycle-events",
                                  json=conv.committed(i, text, answer, now + timedelta(minutes=i)))
            print("  AnswerCommitted ->", r.status_code)
        await asyncio.sleep(1)
        r = await client.get(f"/internal/ai/v1/conversations/{conv.conversation_id}/state",
                             params={"tenantId": conv.tenant_id, "epoch": conv.epoch})
        print("\nstate:", r.json())


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock BE for the AI service")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="replay a short conversation over HTTP + SSE")
    d.add_argument("--base-url", default="http://localhost:8080")
    d.add_argument("--token", default="local-dev-token")
    f = sub.add_parser("fixtures", help="print example payloads as JSON")
    f.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.cmd == "demo":
        asyncio.run(demo(args.base_url, args.token))
    else:
        print(json.dumps(example_fixtures(), ensure_ascii=False, indent=2))


def example_fixtures() -> dict[str, Any]:
    conv = MockConversation(conversation_id="conv-3", character_id="char-9")
    t = datetime(2026, 11, 13, 23, 10, tzinfo=UTC)
    return {
        "ConversationOpened": conv.opened(),
        "AnswerCommitted": conv.committed(11, "나 다음 주 금요일에 면접 있어", "...면접? 잘 봐.", t - timedelta(days=7)),
        "GenerationRequest.NEW": conv.generation(12, "좋은 아침", t),
        "GenerationRequest.REGENERATE": conv.generation(12, "좋은 아침", t, mode="REGENERATE", attempt="b",
                                                        snapshot_id="snap-uuid"),
        "CharacterRevisionChanged": conv.revision_changed(HYUNWOO | {"background": "현재 대학원생이다."}),
        "ConversationReset": conv.reset(),
        "CharacterDeleted": conv.character_deleted(),
    }


if __name__ == "__main__":
    main()
