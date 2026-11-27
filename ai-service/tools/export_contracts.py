"""Regenerate contracts/ (OpenAPI, JSON Schemas, fixtures) from the Pydantic models.

    python -m tools.export_contracts          # write files
    python -m tools.export_contracts --check  # CI: fail if contracts/ is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from app.api import schemas
from app.main import create_app
from tools.mock_be import example_fixtures

ROOT = Path(__file__).resolve().parents[1] / "contracts"


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def build() -> dict[Path, str]:
    files: dict[Path, str] = {ROOT / "openapi.json": _dump(create_app().openapi())}
    models = {
        "GenerationRequest": schemas.GenerationRequest,
        "GenerationAccepted": schemas.GenerationAccepted,
        "GenerationResult": schemas.GenerationResult,
        "LifecycleAccepted": schemas.LifecycleAccepted,
        "LifecycleReceipt": schemas.LifecycleReceipt,
        "ConversationState": schemas.ConversationState,
        "ErrorBody": schemas.ErrorBody,
    }
    for name, model in models.items():
        files[ROOT / "schemas" / f"{name}.schema.json"] = _dump(model.model_json_schema(by_alias=True))
    files[ROOT / "schemas" / "LifecycleEvent.schema.json"] = _dump(
        TypeAdapter(schemas.LifecycleEvent).json_schema(by_alias=True)
    )
    for name, payload in example_fixtures().items():
        files[ROOT / "fixtures" / f"{name}.json"] = _dump(_stable(payload))
    return files


def _stable(payload: dict[str, Any]) -> dict[str, Any]:
    """Fixtures must not change on every run: pin random ids / timestamps."""
    p = json.loads(json.dumps(payload))
    if "eventId" in p:
        p["eventId"] = f"evt-{p['type']}-{p['aggregateSeq']}"
        p["occurredAt"] = "2026-11-13T23:10:00Z"
    if "deadlineAt" in p:
        p["deadlineAt"] = "2026-11-13T23:11:30Z"
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for path, content in build().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if stale:
        print("contracts/ is stale, run `python -m tools.export_contracts`:\n  " + "\n  ".join(stale))
        sys.exit(1)


if __name__ == "__main__":
    main()
