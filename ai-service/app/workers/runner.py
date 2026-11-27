"""Worker process: lifecycle loop + generation slots + memory slots in one process (plan §12 MVP).

Generation and memory slots are separate so a burst of extraction work never takes the
slots users are waiting on. In PROD the same code runs as separate pools.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket

from app.dialogue.manager import DialogueManager
from app.lifecycle.handler import process_pending
from app.memory.pipeline import MemoryPipeline
from app.memory.reindex import reindex_all, stale_counts
from app.persistence import jobs
from app.services import Services

log = logging.getLogger(__name__)


class WorkerRunner:
    def __init__(self, services: Services) -> None:
        self.sv = services
        self.cfg = services.settings
        self.dialogue = DialogueManager(services)
        self.memory = MemoryPipeline(services)
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def _sleep(self) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=self.cfg.poll_interval_seconds)
        except TimeoutError:
            pass

    async def lifecycle_loop(self) -> None:
        while not self._stop.is_set():
            try:
                n = await process_pending(self.sv.db, self.cfg.lifecycle_batch, self.cfg.tuning)
            except Exception:
                log.exception("lifecycle loop error")
                n = 0
            if n == 0:
                await self._sleep()

    async def reindex_loop(self) -> None:
        """Background re-embedding of stale memories (embedding model change, outages)."""
        interval = self.cfg.reindex_interval_seconds
        while not self._stop.is_set():
            try:
                pending = (await stale_counts(self.sv.db, self.cfg.embedding_model)).remaining
                if pending:
                    result = await reindex_all(self.sv.db, self.sv.provider, self.cfg.embedding_model,
                                               self.cfg.reindex_sweep_batch, max_batches=10)
                    log.info("embedding re-index: %d embedded, %d remaining", result.embedded, result.remaining)
            except Exception:
                log.exception("re-index sweep error")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except TimeoutError:
                pass

    async def generation_slot(self, slot: int) -> None:
        owner = f"{self.worker_id}:gen{slot}"
        while not self._stop.is_set():
            job = await self._claim("GENERATION", owner)
            if job is None:
                await self._sleep()
                continue
            await self.dialogue.run_attempt(job)

    async def memory_slot(self, slot: int) -> None:
        owner = f"{self.worker_id}:mem{slot}"
        kinds = ("MEMORY", "REVISION", "BASELINE")
        while not self._stop.is_set():
            job = None
            for kind in kinds:
                job = await self._claim(kind, owner)
                if job:
                    break
            if job is None:
                await self._sleep()
                continue
            await self.memory.run(job)

    async def _claim(self, kind: str, owner: str):
        try:
            return await jobs.claim(self.sv.db, kind, owner, self.cfg.lease_seconds)
        except Exception:
            log.exception("claim failed", extra={"kind": kind})
            return None

    async def run(self) -> None:
        tasks = [asyncio.create_task(self.lifecycle_loop())]
        tasks += [asyncio.create_task(self.generation_slot(i)) for i in range(self.cfg.generation_slots)]
        tasks += [asyncio.create_task(self.memory_slot(i)) for i in range(self.cfg.memory_slots)]
        if self.cfg.reindex_interval_seconds > 0 and self.cfg.reindex_sweep_batch > 0:
            tasks.append(asyncio.create_task(self.reindex_loop()))
        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def main() -> None:
    import signal

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    services = Services.create()
    runner = WorkerRunner(services)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, runner.stop)
    log.info("worker started", extra={"worker": runner.worker_id})
    try:
        await runner.run()
    finally:
        await services.db.dispose()


if __name__ == "__main__":
    asyncio.run(main())
