"""Tests for the dashboard generator."""

from __future__ import annotations

from scripts.build_dashboard import build_html, poison_examples

LLM = {
    "model": "m<b>", "temperature": 0.0,
    "overall": {"samples": 2, "attack_flip_rate": 0.5, "on_task_rate": 1.0},
    "per_category": {
        "vulnerable_code": {"samples": 2, "attack_flip_rate": 0.5,
                            "rule_scope": "code"},
        "misleading_code": {"samples": 1, "attack_flip_rate": 0.0,
                            "rule_scope": "code"},
    },
}


def test_renders_without_any_results() -> None:
    page = build_html(None, None, [])

    assert "Problem statement" in page
    assert page.count("Not run") == 6
    assert "Would poisoned chunks be retrieved?" in page


def test_defense_report_renders_real_numbers() -> None:
    page = build_html(None, None, [], defense={
        "samples": 32,
        "per_category": {"vulnerable_code": {"block_rate": 1.0, "false_positive_rate": 0.0}},
    })
    assert "100%" in page and "0%" in page


def test_shows_real_numbers_and_escapes_html() -> None:
    page = build_html(None, LLM, [])

    assert "50%" in page and "vulnerable_code" in page
    assert "m<b>" not in page and "m&lt;b&gt;" in page


def test_lists_categories_with_effect_only() -> None:
    page = build_html(None, LLM, [])

    assert "Categories with any effect: <b>vulnerable_code</b>" in page


def test_examples_cover_all_seven_categories() -> None:
    rows = poison_examples()

    assert len(rows) == 7 and all(added for _, added in rows)
