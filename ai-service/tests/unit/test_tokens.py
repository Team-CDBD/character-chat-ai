import math

import pytest

from app.config import Settings
from app.providers.base import ChatParams, PromptMessage, ProviderAdapter, Usage
from app.providers.registry import build_counter, check_context_fits
from app.providers.tokens import (
    DEFAULT_CHARS_PER_TOKEN,
    SAFETY_MARGIN,
    CalibratedEstimator,
    TokenCounter,
    estimate_tokens,
)

TEXT = "오늘 면접 준비 때문에 정신이 하나도 없어. " * 20  # 500 chars


def test_estimate_until_calibrated_then_learned_ratio():
    counter = TokenCounter()
    assert counter.source == "estimate" and counter.count(TEXT) == estimate_tokens(TEXT)
    # provider says the prompt was 200 content tokens (+4 template tokens): 2.5 chars/token
    counter.observe([TEXT], 204)
    assert counter.source == "calibrated"
    assert counter.count(TEXT) == math.ceil(len(TEXT) / (2.5 / SAFETY_MARGIN))


def test_cache_shortened_counts_do_not_make_counting_optimistic():
    est = CalibratedEstimator()
    for _ in range(8):
        est.observe([TEXT], 254)  # ~2 chars/token: full prompt evaluated
    for _ in range(2):
        est.observe([TEXT], 104)  # prefix served from cache: looks like 5 chars/token
    assert est.ratio() == pytest.approx(len(TEXT) / 250 / SAFETY_MARGIN)


def test_implausible_and_tiny_samples_are_ignored():
    est = CalibratedEstimator()
    est.observe(["짧음"], 3)
    est.observe([TEXT], 5)  # 100 chars/token: cached or broken report
    est.observe([TEXT], 4)
    assert est.samples == 0 and est.ratio() == DEFAULT_CHARS_PER_TOKEN


class _Reporting:
    name = "ollama"

    def count_tokens(self, text):
        return estimate_tokens(text)

    async def complete_json(self, messages, params, schema_name):
        return {}, Usage(params.model, 208, 1)


async def test_adapter_learns_only_from_the_chat_model():
    adapter = ProviderAdapter(_Reporting(), timeout_seconds=5, max_concurrency=1, counter=TokenCounter(),
                              count_model="gemma4:12b")
    msgs = [PromptMessage("system", TEXT[:250]), PromptMessage("user", TEXT[:250])]
    await adapter.complete_json(msgs, ChatParams(model="other-model"), "x")
    assert adapter.token_count_source == "estimate"
    await adapter.complete_json(msgs, ChatParams(model="gemma4:12b"), "x")
    assert adapter.token_count_source == "calibrated"
    assert adapter.count_tokens(TEXT) == math.ceil(len(TEXT) / (500 / 200 / SAFETY_MARGIN))


def test_hf_tokenizer_file_gives_exact_counts(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tok = tokenizers.Tokenizer(WordLevel({"[UNK]": 0, "안녕": 1, "하세요": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    counter = build_counter(Settings(provider="ollama", tokenizer_file=str(path)), _Reporting())
    assert counter.source == "exact:hf" and counter.count("안녕 하세요 반가워") == 3 and counter.count("") == 0


def test_mock_provider_never_calibrates():
    class Mock(_Reporting):
        name = "mock"

    counter = build_counter(Settings(provider="mock"), Mock())
    counter.observe([TEXT], 204)
    assert counter.source == "estimate"


def test_budget_must_fit_ollama_context():
    check_context_fits(Settings(provider="ollama"))
    with pytest.raises(ValueError, match="AI_OLLAMA_NUM_CTX"):
        check_context_fits(Settings(provider="ollama", ollama_num_ctx=4096))
