"""Re-embed memories after an embedding model change (or an embedding outage).

    python -m tools.reindex_embeddings --status      # what is stale, by version / status
    python -m tools.reindex_embeddings               # re-embed everything stale with AI_EMBEDDING_MODEL
    python -m tools.reindex_embeddings --batch 64 --max-batches 10

Uses the same settings as the worker (AI_PROVIDER, AI_EMBEDDING_MODEL, AI_DATABASE_URL). Safe to run
while the service is up: rows deleted or already re-embedded meanwhile are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import get_settings
from app.memory.reindex import reindex_all, stale_counts
from app.services import Services


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="only show stale counts")
    parser.add_argument("--batch", type=int, default=None, help="memories per embedding call")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)
    settings = get_settings()
    services = Services.create(settings)
    model = settings.embedding_model
    try:
        before = await stale_counts(services.db, model)
        print(f"embedding model: {model}")
        print(f"stale memories: {before.remaining}")
        for key, n in before.by_version.items():
            print(f"  {key}: {n}")
        if args.status or not before.remaining:
            return 0
        result = await reindex_all(services.db, services.provider, model, args.batch or settings.reindex_sweep_batch,
                                   max_batches=args.max_batches)
        print(f"embedded: {result.embedded}, failed batch rows: {result.failed}, remaining: {result.remaining}")
        return 1 if result.failed else 0
    finally:
        await services.db.dispose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
