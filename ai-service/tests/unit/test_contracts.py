import json
from pathlib import Path

from pydantic import TypeAdapter

from app.api.schemas import GenerationRequest, LifecycleEvent

FIXTURES = Path(__file__).resolve().parents[2] / "contracts" / "fixtures"


def test_fixtures_match_contract_models():
    adapter = TypeAdapter(LifecycleEvent)
    files = sorted(FIXTURES.glob("*.json"))
    assert files
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if f.name.startswith("GenerationRequest"):
            GenerationRequest.model_validate(data)
        else:
            assert adapter.validate_python(data).type == data["type"]


def test_contracts_are_up_to_date():
    from tools.export_contracts import build

    for path, content in build().items():
        assert path.read_text(encoding="utf-8") == content, f"{path.name} is stale: run python -m tools.export_contracts"
