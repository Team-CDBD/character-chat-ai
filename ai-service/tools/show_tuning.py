"""Print the effective tuning values (defaults + AI_TUNING_FILE + AI_TUNING__* env) as JSON.

    python -m tools.show_tuning
"""

from __future__ import annotations

import json

from app.config import get_settings


def main() -> None:
    tuning = get_settings().tuning
    print(json.dumps({"version": tuning.version(), **tuning.model_dump(mode="json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
