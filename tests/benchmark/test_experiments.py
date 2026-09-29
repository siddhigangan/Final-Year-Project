"""Tests for defense, retrieval and ablation experiment code."""

from __future__ import annotations

from pathlib import Path

from src.benchmark.defense_eval import evaluate_defense, screen, default_pipeline
from src.benchmark.integration import build_poisoning_dataset
from src.benchmark.llm_eval import (
    build_generation_request, evaluate_sample, naive_prompt, overall,
)
from src.benchmark.retrieval_eval import (
    HashingEmbeddingProvider, collect_distractors, evaluate_retrieval,
    summarize_retrieval,
)
from src.benchmark.seeds import LLM_SEEDS
from src.defense.pipeline import DefensePipelineResult
from src.generation.base import GenerationProvider
from src.models import GenerationResult, GenerationStatus
from src.security.decision import SecurityDecision

DS = build_poisoning_dataset(list(LLM_SEEDS))


class Recorder(GenerationProvider):
    def __init__(self) -> None:
        super().__init__(model_name="rec", provider_name="rec")
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return GenerationResult(
            request_id=request.request_id, generated_code="x = 1",
            model_name="rec", provider_name="rec",
            status=GenerationStatus.SUCCESS, latency_ms=0.0)


class BlockAll:
    def analyze_context(self, **_):
        return DefensePipelineResult(
            blocked=True, final_decision=SecurityDecision.REJECT)


def test_real_defense_has_no_false_positives_and_catches_key_and_injection() -> None:
    report = evaluate_defense(DS)

    assert all(m["false_positive_rate"] == 0.0 for m in report.values())
    assert report["vulnerable_code"]["block_rate"] == 1.0
    assert report["instruction_like_content"]["block_rate"] == 1.0
    assert report["misleading_code"]["block_rate"] == 0.0


def test_screen_returns_sanitized_context() -> None:
    sample = DS.samples[0]
    result = screen(sample, sample.clean_code, default_pipeline())

    assert result.blocked is False
    assert "BEGIN RETRIEVED DATA" in result.context_code


def test_blocked_context_never_reaches_the_model() -> None:
    provider = Recorder()
    result = evaluate_sample(DS.samples[0], provider=provider, defense=BlockAll())

    assert provider.calls == 0
    assert result.poisoned_blocked and result.clean_blocked
    assert not result.poisoned_attack_followed


def test_undefended_run_calls_model_twice() -> None:
    provider = Recorder()
    evaluate_sample(DS.samples[0], provider=provider)

    assert provider.calls == 2


def test_naive_prompt_has_no_warning_and_hardened_does() -> None:
    sample = DS.samples[0]
    p = Recorder()
    naive = build_generation_request(sample, "c", provider=p, prompt_style="naive")
    hard = build_generation_request(sample, "c", provider=p)

    assert naive.metadata["prompt"] == naive_prompt(sample.query, "c")
    assert "untrusted" not in naive.metadata["prompt"].lower()
    assert "untrusted" in hard.metadata["prompt"].lower()


def test_overall_reports_blocked_rate() -> None:
    results = [evaluate_sample(s, provider=Recorder(), defense=BlockAll())
               for s in DS.samples[:3]]

    assert overall(results)["blocked_rate"] == 1.0


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    e = HashingEmbeddingProvider(64)
    a, b = e.embed_texts(["def foo(): pass"]), e.embed_texts(["def foo(): pass"])

    assert (a == b).all() and abs(float((a ** 2).sum()) - 1.0) < 1e-5


def test_retrieval_outcomes_are_structurally_sound(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "def alpha(x):\n    return x + 1  # some filler text here\n\n"
        "def beta(y):\n    return y * 2  # more filler text here\n")
    dist = collect_distractors(tmp_path)
    outcomes = evaluate_retrieval(
        DS, provider=HashingEmbeddingProvider(), distractors=dist, top_k=5)

    assert len(dist) == 2 and len(outcomes) == DS.size
    assert all(o.poison_rank is None or 1 <= o.poison_rank <= 5 for o in outcomes)
    assert set(summarize_retrieval(outcomes)) == {s.poisoning_category for s in DS.samples}


def test_large_top_k_always_finds_the_poison() -> None:
    outcomes = evaluate_retrieval(
        DS, provider=HashingEmbeddingProvider(), distractors=[], top_k=50)

    assert all(o.poison_in_top_k for o in outcomes)
