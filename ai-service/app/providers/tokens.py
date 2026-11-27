"""Token counting for context budgets (plan §06, §12).

Order of preference, picked once at startup and reported as ``TokenCounter.source``:

1. ``exact:hf``       - a Hugging Face ``tokenizer.json`` for the chat model (``AI_TOKENIZER_FILE``),
                        e.g. the Gemma tokenizer. Needs the optional ``tokenizers`` package.
2. ``exact:tiktoken`` - OpenAI models when the optional ``tiktoken`` package and its encoding file are
                        available.
3. ``calibrated``     - a characters-per-token ratio learned from the prompt token counts the provider
                        reports on real calls (Ollama ``prompt_eval_count``, OpenAI ``usage``).
4. ``estimate``       - the fixed conservative ratio (1.5 characters per token) until calibration has
                        samples, and always for the mock provider.

The calibrated ratio deliberately leans high on token count: it uses a low percentile of recent
samples, so a call whose prompt prefix was served from the provider's cache (which reports fewer
prompt tokens) cannot make later contexts overflow, and then applies a safety margin.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Callable, Sequence

log = logging.getLogger(__name__)

DEFAULT_CHARS_PER_TOKEN = 1.5
PER_MESSAGE_OVERHEAD = 4  # chat template tokens per message (role markers, separators)
MIN_SAMPLE_CHARS = 200  # tiny prompts are dominated by template overhead
PLAUSIBLE_RATIO = (0.5, 6.0)
WINDOW = 50
PERCENTILE = 0.2
SAFETY_MARGIN = 1.1


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / chars_per_token)


class CalibratedEstimator:
    def __init__(self, default_ratio: float = DEFAULT_CHARS_PER_TOKEN) -> None:
        self.default_ratio = default_ratio
        self._samples: deque[float] = deque(maxlen=WINDOW)
        self._lock = threading.Lock()

    @property
    def samples(self) -> int:
        return len(self._samples)

    def ratio(self) -> float:
        with self._lock:
            if not self._samples:
                return self.default_ratio
            ordered = sorted(self._samples)
        low = ordered[min(len(ordered) - 1, int(len(ordered) * PERCENTILE))]
        return low / SAFETY_MARGIN

    def observe(self, texts: Sequence[str], reported_input_tokens: int) -> None:
        chars = sum(len(t) for t in texts)
        content_tokens = reported_input_tokens - PER_MESSAGE_OVERHEAD * len(texts)
        if chars < MIN_SAMPLE_CHARS or content_tokens <= 0:
            return
        sample = chars / content_tokens
        if not PLAUSIBLE_RATIO[0] <= sample <= PLAUSIBLE_RATIO[1]:
            return
        with self._lock:
            self._samples.append(sample)

    def count(self, text: str) -> int:
        return estimate_tokens(text, self.ratio())


def load_hf_tokenizer(path: str) -> Callable[[str], int]:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ValueError("AI_TOKENIZER_FILE needs the 'tokenizers' package: pip install '.[tokenizer]'") from exc
    tok = Tokenizer.from_file(path)

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids) if text else 0

    return count


def load_tiktoken(model: str) -> Callable[[str], int] | None:
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        enc.encode("probe")
    except Exception:  # encoding file not cached and not downloadable
        log.warning("tiktoken encoding unavailable; falling back to calibrated estimate")
        return None

    def count(text: str) -> int:
        return len(enc.encode(text, disallowed_special=())) if text else 0

    return count


class TokenCounter:
    def __init__(
        self,
        *,
        exact: Callable[[str], int] | None = None,
        exact_source: str = "exact",
        calibrate: bool = True,
        fallback: Callable[[str], int] = estimate_tokens,
    ) -> None:
        self._exact = exact
        self._calibrated = CalibratedEstimator() if calibrate and exact is None else None
        self._fallback = fallback
        self._exact_source = exact_source

    @property
    def source(self) -> str:
        if self._exact is not None:
            return self._exact_source
        if self._calibrated is not None and self._calibrated.samples:
            return "calibrated"
        return "estimate"

    def count(self, text: str) -> int:
        if self._exact is not None:
            return self._exact(text)
        if self._calibrated is not None and self._calibrated.samples:
            return self._calibrated.count(text)
        return self._fallback(text)

    def observe(self, texts: Sequence[str], reported_input_tokens: int) -> None:
        if self._calibrated is not None:
            self._calibrated.observe(texts, reported_input_tokens)
