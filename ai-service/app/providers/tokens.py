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

The calibrated ratio deliberately leans high on token count. Ollama reports only the prompt tokens it
had to evaluate, so a call whose prefix came from the KV cache looks like "more characters per token".
To keep such samples from making later contexts overflow ``num_ctx``:

* calibration starts only after ``MIN_SAMPLES`` samples (the fixed estimate is used until then);
* it uses a low percentile of recent samples, then a safety margin;
* the learned ratio can never exceed ``MAX_RATIO_FACTOR`` x the default ratio;
* samples whose reported count is at the context limit (a truncated prompt) are dropped.

For production use an exact tokenizer (``AI_TOKENIZER_FILE``); calibration is the fallback.
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
MIN_SAMPLES = 10
MAX_RATIO_FACTOR = 1.2  # never count fewer tokens than default_ratio * 1.2 chars/token would


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / chars_per_token)


class CalibratedEstimator:
    def __init__(self, default_ratio: float = DEFAULT_CHARS_PER_TOKEN, context_limit: int | None = None) -> None:
        self.default_ratio = default_ratio
        self.context_limit = context_limit
        self._samples: deque[float] = deque(maxlen=WINDOW)
        self._lock = threading.Lock()

    @property
    def samples(self) -> int:
        return len(self._samples)

    @property
    def ready(self) -> bool:
        return len(self._samples) >= MIN_SAMPLES

    def ratio(self) -> float:
        with self._lock:
            if len(self._samples) < MIN_SAMPLES:
                return self.default_ratio
            ordered = sorted(self._samples)
        low = ordered[min(len(ordered) - 1, int(len(ordered) * PERCENTILE))]
        return min(low / SAFETY_MARGIN, self.default_ratio * MAX_RATIO_FACTOR)

    def observe(self, texts: Sequence[str], reported_input_tokens: int) -> None:
        chars = sum(len(t) for t in texts)
        content_tokens = reported_input_tokens - PER_MESSAGE_OVERHEAD * len(texts)
        if chars < MIN_SAMPLE_CHARS or content_tokens <= 0:
            return
        if self.context_limit and reported_input_tokens >= self.context_limit - 16:
            return  # the provider truncated the prompt: the count says nothing about the text
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
        context_limit: int | None = None,
    ) -> None:
        self._exact = exact
        self._calibrated = CalibratedEstimator(context_limit=context_limit) if calibrate and exact is None else None
        self._fallback = fallback
        self._exact_source = exact_source

    @property
    def source(self) -> str:
        if self._exact is not None:
            return self._exact_source
        if self._calibrated is not None and self._calibrated.ready:
            return "calibrated"
        return "estimate"

    def count(self, text: str) -> int:
        if self._exact is not None:
            return self._exact(text)
        if self._calibrated is not None and self._calibrated.ready:
            return self._calibrated.count(text)
        return self._fallback(text)

    def observe(self, texts: Sequence[str], reported_input_tokens: int) -> None:
        if self._calibrated is not None:
            self._calibrated.observe(texts, reported_input_tokens)
