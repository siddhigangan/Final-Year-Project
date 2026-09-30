"""Tests for the defense layer ablation (blueprint section 32)."""

from __future__ import annotations

from src.benchmark.integration import build_poisoning_dataset
from src.benchmark.layer_ablation import (
    LADDER,
    evaluate_ablation,
    evaluate_layer_ablation,
    evaluate_layer_ablation_by_category,
)
from src.benchmark.seeds import LLM_SEEDS
from src.defense.pipeline import DefensePipeline, DefensePipelineConfig

DS = build_poisoning_dataset(list(LLM_SEEDS))


def test_ladder_has_six_steps_in_spec_order() -> None:
    assert LADDER == (
        "no_defense", "L1", "L1_L2", "L1_L2_L3", "L1_L2_L3_L4", "L1_L2_L3_L4_L5",
    )


def test_no_defense_step_never_blocks() -> None:
    for sample in DS.samples:
        result = evaluate_ablation(
            sample,
            pipeline=DefensePipeline(config=DefensePipelineConfig(enabled=True)),
        )
        assert result.step("no_defense").blocked is False
        assert result.step("no_defense").finding_count == 0


def test_block_rate_never_decreases_along_the_ladder() -> None:
    summary = evaluate_layer_ablation(DS)
    rates = [summary[step]["block_rate"] for step in LADDER]

    assert rates == sorted(rates)


def test_l3_step_matches_l2_step_exactly() -> None:
    # L3 (context validation) never emits a SecurityFinding in the
    # current pipeline, so adding it to the ladder must not change the
    # block decision for any sample.
    for sample in DS.samples:
        result = evaluate_ablation(
            sample,
            pipeline=DefensePipeline(config=DefensePipelineConfig(enabled=True)),
        )
        assert result.step("L1_L2").blocked == result.step("L1_L2_L3").blocked


def test_vulnerable_code_is_caught_only_at_the_final_l5_step() -> None:
    by_category = evaluate_layer_ablation_by_category(DS)
    rates = [
        by_category["vulnerable_code"][step]["block_rate"] for step in LADDER
    ]

    assert rates == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_instruction_like_content_is_caught_starting_at_l2() -> None:
    by_category = evaluate_layer_ablation_by_category(DS)
    rates = [
        by_category["instruction_like_content"][step]["block_rate"]
        for step in LADDER
    ]

    assert rates == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_five_categories_are_never_blocked_at_any_step() -> None:
    by_category = evaluate_layer_ablation_by_category(DS)
    never_blocked = {
        "misleading_code", "false_api_guidance", "contradictory_documentation",
        "false_repository_conventions", "context_manipulation",
    }

    for category in never_blocked:
        rates = [by_category[category][step]["block_rate"] for step in LADDER]
        assert rates == [0.0] * len(LADDER)


def test_by_category_and_overall_agree() -> None:
    overall = evaluate_layer_ablation(DS)
    by_category = evaluate_layer_ablation_by_category(DS)

    for step in LADDER:
        total_blocked = sum(
            by_category[cat][step]["block_rate"] * by_category[cat][step]["samples"]
            for cat in by_category
        )
        total_samples = sum(
            by_category[cat][step]["samples"] for cat in by_category
        )
        assert abs(total_blocked / total_samples - overall[step]["block_rate"]) < 1e-9
