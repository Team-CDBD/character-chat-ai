"""Prompt Registry: versioned prompt files under ``prompts/``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
CHAT_VERSIONS = {"chat-v1"}


@lru_cache
def load(name: str) -> str:
    path = PROMPT_DIR / f"{name}.txt"
    if not path.is_file():
        raise KeyError(f"unknown prompt: {name}")
    return path.read_text(encoding="utf-8").strip()


def chat_policy(version: str) -> str:
    if version not in CHAT_VERSIONS:
        raise KeyError(f"unknown generation policy version: {version}")
    return load(f"{version}.policy")
