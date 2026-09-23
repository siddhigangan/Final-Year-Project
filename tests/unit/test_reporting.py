"""Unit tests for SecureCodeRAG experiment reporting."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.experiments.reporting import (
    ConditionComparison,
    ConditionMetrics,
    ExperimentReport,
    ExperimentReporter,
    ReportingInputError,
    SecurityUtilityAnalysis,
)
from src.models import (
    ExperimentCondition,
    ExperimentResult,
    SecurityDecision,
    SecurityFinding,
    SecuritySeverity,
)


def make_result(
    *,
    experiment_id: str,
    condition: ExperimentCondition,
    poison_retrieval_rate: float = 0.0,
    attack_success_rate: float = 0.0,
    vulnerability_introduction_rate: float = 0.0,
    utility_score: float = 0.8,
    latency_ms: float = 100.0,
    generation_changed: bool = False,
    vulnerability_introduced: bool = False,
    findings: list[SecurityFinding] | None = None,
    repository_id: str = "repo-001",
    task_type: str = "code_generation",
    model_name: str = "test-model",
) -> ExperimentResult:
    """Create a valid ExperimentResult for reporting tests."""
    return ExperimentResult(
        experiment_id=experiment_id,
        condition=condition,
        request_id=f"request-{experiment_id}",
        poison_present=condition
        in {
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
        },
        poison_retrieved=poison_retrieval_rate > 0.0,
        poison_in_context=poison_retrieval_rate > 0.0,
        generation_changed=generation_changed,
        vulnerability_introduced=vulnerability_introduced,
        poison_retrieval_rate=poison_retrieval_rate,
        attack_success_rate=attack_success_rate,
        vulnerability_introduction_rate=(
            vulnerability_introduction_rate
        ),
        utility_score=utility_score,
        latency_ms=latency_ms,
        findings=findings or [],
        metadata={
            "repository_id": repository_id,
            "task_type": task_type,
            "model_name": model_name,
            "retriever_name": "test-retriever",
            "defense_enabled": condition
            in {
                ExperimentCondition.POISONED_DEFENSE,
                ExperimentCondition.CLEAN_DEFENSE,
            },
        },
        created_at=datetime(
            2026,
            9,
            23,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )


def make_finding(
    *,
    finding_id: str,
    decision: SecurityDecision = SecurityDecision.FLAG,
) -> SecurityFinding:
    """Create a valid model-level SecurityFinding."""
    return SecurityFinding(
        finding_id=finding_id,
        rule_id=f"RULE-{finding_id}",
        title="Test security finding",
        description="Synthetic finding used for unit testing.",
        severity=SecuritySeverity.MEDIUM,
        decision=decision,
        file_path="test.py",
        start_line=1,
        end_line=1,
        evidence="synthetic evidence",
        analyzer="test-analyzer",
        confidence=0.9,
        metadata={},
    )


def make_all_conditions() -> list[ExperimentResult]:
    """Create one representative result for each A/B/C/D condition."""
    return [
        make_result(
            experiment_id="A-001",
            condition=ExperimentCondition.CLEAN_NO_DEFENSE,
            poison_retrieval_rate=0.0,
            attack_success_rate=0.0,
            vulnerability_introduction_rate=0.0,
            utility_score=0.90,
            latency_ms=100.0,
        ),
        make_result(
            experiment_id="B-001",
            condition=ExperimentCondition.POISONED_NO_DEFENSE,
            poison_retrieval_rate=0.80,
            attack_success_rate=0.60,
            vulnerability_introduction_rate=0.40,
            utility_score=0.85,
            latency_ms=110.0,
            generation_changed=True,
            vulnerability_introduced=True,
        ),
        make_result(
            experiment_id="C-001",
            condition=ExperimentCondition.POISONED_DEFENSE,
            poison_retrieval_rate=0.20,
            attack_success_rate=0.10,
            vulnerability_introduction_rate=0.05,
            utility_score=0.82,
            latency_ms=130.0,
        ),
        make_result(
            experiment_id="D-001",
            condition=ExperimentCondition.CLEAN_DEFENSE,
            poison_retrieval_rate=0.0,
            attack_success_rate=0.0,
            vulnerability_introduction_rate=0.0,
            utility_score=0.87,
            latency_ms=125.0,
        ),
    ]


class TestConditionMetrics:
    """Tests for condition-level aggregate metrics."""

    def test_condition_properties(
        self,
    ) -> None:
        clean = ConditionMetrics(
            condition=ExperimentCondition.CLEAN_NO_DEFENSE,
            sample_count=1,
            poison_retrieval_rate=0.0,
            attack_success_rate=0.0,
            vulnerability_introduction_rate=0.0,
            utility_score=0.9,
            latency_ms=100.0,
            generation_change_rate=0.0,
            vulnerability_rate=0.0,
            security_finding_count=0,
            blocking_finding_count=0,
        )

        poisoned_defended = ConditionMetrics(
            condition=ExperimentCondition.POISONED_DEFENSE,
            sample_count=1,
            poison_retrieval_rate=0.1,
            attack_success_rate=0.1,
            vulnerability_introduction_rate=0.05,
            utility_score=0.8,
            latency_ms=120.0,
            generation_change_rate=0.1,
            vulnerability_rate=0.05,
            security_finding_count=1,
            blocking_finding_count=1,
        )

        assert clean.defense_enabled is False
        assert clean.poisoned is False

        assert poisoned_defended.defense_enabled is True
        assert poisoned_defended.poisoned is True

    def test_condition_metrics_as_dict(
        self,
    ) -> None:
        metrics = ConditionMetrics(
            condition=ExperimentCondition.POISONED_NO_DEFENSE,
            sample_count=2,
            poison_retrieval_rate=0.5,
            attack_success_rate=0.4,
            vulnerability_introduction_rate=0.2,
            utility_score=0.8,
            latency_ms=110.0,
            generation_change_rate=0.5,
            vulnerability_rate=0.5,
            security_finding_count=2,
            blocking_finding_count=1,
        )

        data = metrics.as_dict()

        assert data["condition"] == "B_poisoned_no_defense"
        assert data["sample_count"] == 2
        assert data["attack_success_rate"] == 0.4
        assert data["defense_enabled"] is False
        assert data["poisoned"] is True


class TestConditionComparison:
    """Tests for condition comparison objects."""

    def test_comparison_as_dict(
        self,
    ) -> None:
        comparison = ConditionComparison(
            baseline=ExperimentCondition.POISONED_NO_DEFENSE,
            comparison=ExperimentCondition.POISONED_DEFENSE,
            poison_retrieval_rate_delta=-0.6,
            attack_success_rate_delta=-0.5,
            vulnerability_introduction_rate_delta=-0.35,
            utility_score_delta=-0.03,
            latency_ms_delta=20.0,
            generation_change_rate_delta=-1.0,
            vulnerability_rate_delta=-1.0,
        )

        data = comparison.as_dict()

        assert (
            data["baseline"]
            == "B_poisoned_no_defense"
        )
        assert (
            data["comparison"]
            == "C_poisoned_defense"
        )
        assert data["attack_success_rate_delta"] == -0.5
        assert data["latency_ms_delta"] == 20.0


class TestSecurityUtilityAnalysisSerialization:
    """Tests for security–utility analysis serialization."""

    def test_analysis_as_dict(
        self,
    ) -> None:
        analysis = SecurityUtilityAnalysis(
            clean_baseline=(
                ExperimentCondition.CLEAN_NO_DEFENSE
            ),
            poisoned_baseline=(
                ExperimentCondition.POISONED_NO_DEFENSE
            ),
            defended_poisoned=(
                ExperimentCondition.POISONED_DEFENSE
            ),
            defended_clean=(
                ExperimentCondition.CLEAN_DEFENSE
            ),
            attack_success_reduction=0.5,
            vulnerability_reduction=0.35,
            poison_retrieval_reduction=0.6,
            defended_poisoned_utility_delta=-0.03,
            defended_clean_utility_delta=-0.03,
            defended_poisoned_latency_delta_ms=20.0,
            defended_clean_latency_delta_ms=25.0,
        )

        data = analysis.as_dict()

        assert (
            data["clean_baseline"]
            == "A_clean_no_defense"
        )
        assert (
            data["defended_poisoned"]
            == "C_poisoned_defense"
        )
        assert data["attack_success_reduction"] == 0.5
        assert data["vulnerability_reduction"] == 0.35


class TestExperimentReporterInitialization:
    """Tests for reporter construction and result management."""

    def test_empty_reporter(
        self,
    ) -> None:
        reporter = ExperimentReporter()

        assert reporter.result_count == 0
        assert reporter.results == ()

    def test_add_result(
        self,
    ) -> None:
        reporter = ExperimentReporter()

        result = make_result(
            experiment_id="exp-001",
            condition=ExperimentCondition.CLEAN_NO_DEFENSE,
        )

        reporter.add_result(result)

        assert reporter.result_count == 1
        assert reporter.results[0] == result

    def test_add_results(
        self,
    ) -> None:
        results = make_all_conditions()

        reporter = ExperimentReporter(results)

        assert reporter.result_count == 4
        assert reporter.results == tuple(results)

    def test_duplicate_experiment_id_is_rejected(
        self,
    ) -> None:
        result = make_result(
            experiment_id="duplicate",
            condition=ExperimentCondition.CLEAN_NO_DEFENSE,
        )

        reporter = ExperimentReporter()

        reporter.add_result(result)

        with pytest.raises(
            ReportingInputError,
            match="Duplicate experiment ID",
        ):
            reporter.add_result(result)

    def test_invalid_result_is_rejected(
        self,
    ) -> None:
        reporter = ExperimentReporter()

        with pytest.raises(
            ReportingInputError,
            match="ExperimentResult",
        ):
            reporter.add_result(
                "invalid",  # type: ignore[arg-type]
            )

    def test_string_iterable_is_rejected(
        self,
    ) -> None:
        with pytest.raises(
            ReportingInputError,
            match="iterable of ExperimentResult",
        ):
            ExperimentReporter(
                "invalid",  # type: ignore[arg-type]
            )

    def test_invalid_metadata_is_rejected(
        self,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="metadata must be a dictionary",
        ):
            ExperimentReporter(
                metadata="invalid",  # type: ignore[arg-type]
            )

    def test_clear(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            [make_all_conditions()[0]]
        )

        reporter.clear()

        assert reporter.result_count == 0
        assert reporter.results == ()

    def test_repr(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            [make_all_conditions()[0]]
        )

        assert repr(reporter) == (
            "ExperimentReporter(results=1)"
        )


class TestConditionAggregation:
    """Tests for aggregate metric calculations."""

    def test_condition_metrics_calculation(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            [
                make_result(
                    experiment_id="exp-001",
                    condition=(
                        ExperimentCondition.POISONED_NO_DEFENSE
                    ),
                    poison_retrieval_rate=0.4,
                    attack_success_rate=0.2,
                    vulnerability_introduction_rate=0.1,
                    utility_score=0.8,
                    latency_ms=100.0,
                    generation_changed=True,
                    vulnerability_introduced=True,
                ),
                make_result(
                    experiment_id="exp-002",
                    condition=(
                        ExperimentCondition.POISONED_NO_DEFENSE
                    ),
                    poison_retrieval_rate=0.8,
                    attack_success_rate=0.6,
                    vulnerability_introduction_rate=0.3,
                    utility_score=0.9,
                    latency_ms=200.0,
                    generation_changed=False,
                    vulnerability_introduced=False,
                ),
            ]
        )

        metrics = reporter.condition_metrics(
            ExperimentCondition.POISONED_NO_DEFENSE
        )

        assert metrics.sample_count == 2
        assert metrics.poison_retrieval_rate == pytest.approx(
            0.6
        )
        assert metrics.attack_success_rate == pytest.approx(
            0.4
        )
        assert (
            metrics.vulnerability_introduction_rate
            == pytest.approx(0.2)
        )
        assert metrics.utility_score == pytest.approx(
            0.85
        )
        assert metrics.latency_ms == pytest.approx(150.0)
        assert metrics.generation_change_rate == pytest.approx(
            0.5
        )
        assert metrics.vulnerability_rate == pytest.approx(
            0.5
        )

    def test_finding_counts_are_aggregated(
        self,
    ) -> None:
        findings = [
            make_finding(
                finding_id="finding-001",
                decision=SecurityDecision.FLAG,
            ),
            make_finding(
                finding_id="finding-002",
                decision=SecurityDecision.REJECT,
            ),
        ]

        reporter = ExperimentReporter(
            [
                make_result(
                    experiment_id="exp-001",
                    condition=(
                        ExperimentCondition.POISONED_DEFENSE
                    ),
                    findings=findings,
                ),
                make_result(
                    experiment_id="exp-002",
                    condition=(
                        ExperimentCondition.POISONED_DEFENSE
                    ),
                    findings=[
                        make_finding(
                            finding_id="finding-003",
                            decision=SecurityDecision.REJECT,
                        )
                    ],
                ),
            ]
        )

        metrics = reporter.condition_metrics(
            ExperimentCondition.POISONED_DEFENSE
        )

        assert metrics.security_finding_count == 3
        assert metrics.blocking_finding_count == 2

    def test_all_condition_metrics_are_in_enum_order(
        self,
    ) -> None:
        results = list(reversed(make_all_conditions()))

        reporter = ExperimentReporter(results)

        metrics = reporter.all_condition_metrics()

        assert [
            item.condition
            for item in metrics
        ] == [
            ExperimentCondition.CLEAN_NO_DEFENSE,
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
            ExperimentCondition.CLEAN_DEFENSE,
        ]

    def test_missing_condition_raises(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            [
                make_result(
                    experiment_id="exp-001",
                    condition=(
                        ExperimentCondition.CLEAN_NO_DEFENSE
                    ),
                )
            ]
        )

        with pytest.raises(
            ReportingInputError,
            match="No experiment results available",
        ):
            reporter.condition_metrics(
                ExperimentCondition.POISONED_DEFENSE
            )

    def test_invalid_condition_type_is_rejected(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            [make_all_conditions()[0]]
        )

        with pytest.raises(
            ReportingInputError,
            match="condition must be an ExperimentCondition",
        ):
            reporter.condition_metrics(
                "A_clean_no_defense"  # type: ignore[arg-type]
            )


class TestConditionComparisons:
    """Tests for condition-level comparisons."""

    def test_compare_poisoned_without_defense_to_defense(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            make_all_conditions()
        )

        comparison = reporter.compare_conditions(
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
        )

        assert (
            comparison.baseline
            == ExperimentCondition.POISONED_NO_DEFENSE
        )
        assert (
            comparison.comparison
            == ExperimentCondition.POISONED_DEFENSE
        )

        assert comparison.poison_retrieval_rate_delta == pytest.approx(
            -0.6
        )
        assert comparison.attack_success_rate_delta == pytest.approx(
            -0.5
        )
        assert (
            comparison.vulnerability_introduction_rate_delta
            == pytest.approx(-0.35)
        )
        assert comparison.utility_score_delta == pytest.approx(
            -0.03
        )
        assert comparison.latency_ms_delta == pytest.approx(
            20.0
        )

    def test_compare_clean_without_defense_to_defense(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            make_all_conditions()
        )

        comparison = reporter.compare_conditions(
            ExperimentCondition.CLEAN_NO_DEFENSE,
            ExperimentCondition.CLEAN_DEFENSE,
        )

        assert comparison.utility_score_delta == pytest.approx(
            -0.03
        )
        assert comparison.latency_ms_delta == pytest.approx(
            25.0
        )

    def test_comparison_requires_available_conditions(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            [
                make_result(
                    experiment_id="exp-001",
                    condition=(
                        ExperimentCondition.CLEAN_NO_DEFENSE
                    ),
                )
            ]
        )

        with pytest.raises(
            ReportingInputError,
            match="No experiment results available",
        ):
            reporter.compare_conditions(
                ExperimentCondition.CLEAN_NO_DEFENSE,
                ExperimentCondition.POISONED_DEFENSE,
            )


class TestSecurityUtilityAnalysis:
    """Tests for A/B/C/D security–utility calculations."""

    def test_complete_analysis(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            make_all_conditions()
        )

        analysis = reporter.security_utility_analysis()

        assert (
            analysis.attack_success_reduction
            == pytest.approx(0.5)
        )
        assert (
            analysis.vulnerability_reduction
            == pytest.approx(0.35)
        )
        assert (
            analysis.poison_retrieval_reduction
            == pytest.approx(0.6)
        )
        assert (
            analysis.defended_poisoned_utility_delta
            == pytest.approx(-0.03)
        )
        assert (
            analysis.defended_clean_utility_delta
            == pytest.approx(-0.03)
        )
        assert (
            analysis.defended_poisoned_latency_delta_ms
            == pytest.approx(20.0)
        )
        assert (
            analysis.defended_clean_latency_delta_ms
            == pytest.approx(25.0)
        )

    def test_analysis_requires_all_four_conditions(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            make_all_conditions()[:3]
        )

        with pytest.raises(
            ReportingInputError,
            match="requires all four conditions",
        ):
            reporter.security_utility_analysis()


class TestExperimentReport:
    """Tests for complete report generation."""

    def test_build_complete_report(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            make_all_conditions(),
            metadata={
                "experiment_group": "test-group",
            },
        )

        report = reporter.build_report()

        assert isinstance(report, ExperimentReport)
        assert report.total_experiments == 4
        assert report.condition_count == 4
        assert report.has_all_conditions is True
        assert report.security_utility is not None
        assert len(report.comparisons) == 3
        assert report.metadata["experiment_group"] == (
            "test-group"
        )

    def test_report_contains_expected_comparisons(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            make_all_conditions()
        )

        report = reporter.build_report()

        pairs = {
            (
                comparison.baseline,
                comparison.comparison,
            )
            for comparison in report.comparisons
        }

        assert pairs == {
            (
                ExperimentCondition.CLEAN_NO_DEFENSE,
                ExperimentCondition.POISONED_NO_DEFENSE,
            ),
            (
                ExperimentCondition.POISONED_NO_DEFENSE,
                ExperimentCondition.POISONED_DEFENSE,
            ),
            (
                ExperimentCondition.CLEAN_NO_DEFENSE,
                ExperimentCondition.CLEAN_DEFENSE,
            ),
        }

    def test_report_total_findings(
        self,
    ) -> None:
        results = make_all_conditions()

        results[2] = make_result(
            experiment_id="C-001",
            condition=(
                ExperimentCondition.POISONED_DEFENSE
            ),
            findings=[
                make_finding(
                    finding_id="finding-001",
                    decision=SecurityDecision.FLAG,
                ),
                make_finding(
                    finding_id="finding-002",
                    decision=SecurityDecision.REJECT,
                ),
            ],
        )

        reporter = ExperimentReporter(results)

        report = reporter.build_report()

        assert report.total_security_findings == 2
        assert report.total_blocking_findings == 1

    def test_partial_report_does_not_create_security_utility_analysis(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            make_all_conditions()[:2]
        )

        report = reporter.build_report()

        assert report.total_experiments == 2
        assert report.security_utility is None
        assert report.has_all_conditions is False

    def test_empty_report(
        self,
    ) -> None:
        reporter = ExperimentReporter()

        report = reporter.build_report()

        assert report.total_experiments == 0
        assert report.conditions == ()
        assert report.comparisons == ()
        assert report.security_utility is None
        assert report.total_security_findings == 0
        assert report.total_blocking_findings == 0

    def test_report_as_dict(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            make_all_conditions()
        )

        data = reporter.build_report().as_dict()

        assert data["total_experiments"] == 4
        assert len(data["conditions"]) == 4
        assert len(data["comparisons"]) == 3
        assert data["security_utility"] is not None
        assert (
            data["security_utility"][
                "attack_success_reduction"
            ]
            == pytest.approx(0.5)
        )

    def test_report_is_json_serializable(
        self,
    ) -> None:
        import json

        reporter = ExperimentReporter(
            make_all_conditions()
        )

        payload = reporter.build_report().as_dict()

        serialized = json.dumps(
            payload,
            sort_keys=True,
        )

        assert isinstance(serialized, str)
        assert "C_poisoned_defense" in serialized


class TestReportingIntegrationBehavior:
    """Tests for realistic reporting scenarios."""

    def test_multiple_samples_are_aggregated_per_condition(
        self,
    ) -> None:
        results = [
            make_result(
                experiment_id="B-001",
                condition=(
                    ExperimentCondition.POISONED_NO_DEFENSE
                ),
                attack_success_rate=0.2,
                utility_score=0.7,
            ),
            make_result(
                experiment_id="B-002",
                condition=(
                    ExperimentCondition.POISONED_NO_DEFENSE
                ),
                attack_success_rate=0.8,
                utility_score=0.9,
            ),
        ]

        reporter = ExperimentReporter(results)

        metrics = reporter.condition_metrics(
            ExperimentCondition.POISONED_NO_DEFENSE
        )

        assert metrics.sample_count == 2
        assert metrics.attack_success_rate == pytest.approx(
            0.5
        )
        assert metrics.utility_score == pytest.approx(0.8)

    def test_generation_change_rate_is_binary_mean(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            [
                make_result(
                    experiment_id="exp-001",
                    condition=(
                        ExperimentCondition.CLEAN_NO_DEFENSE
                    ),
                    generation_changed=True,
                ),
                make_result(
                    experiment_id="exp-002",
                    condition=(
                        ExperimentCondition.CLEAN_NO_DEFENSE
                    ),
                    generation_changed=False,
                ),
                make_result(
                    experiment_id="exp-003",
                    condition=(
                        ExperimentCondition.CLEAN_NO_DEFENSE
                    ),
                    generation_changed=True,
                ),
                make_result(
                    experiment_id="exp-004",
                    condition=(
                        ExperimentCondition.CLEAN_NO_DEFENSE
                    ),
                    generation_changed=False,
                ),
            ]
        )

        metrics = reporter.condition_metrics(
            ExperimentCondition.CLEAN_NO_DEFENSE
        )

        assert metrics.generation_change_rate == pytest.approx(
            0.5
        )

    def test_vulnerability_rate_is_binary_mean(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            [
                make_result(
                    experiment_id="exp-001",
                    condition=(
                        ExperimentCondition.POISONED_DEFENSE
                    ),
                    vulnerability_introduced=True,
                ),
                make_result(
                    experiment_id="exp-002",
                    condition=(
                        ExperimentCondition.POISONED_DEFENSE
                    ),
                    vulnerability_introduced=False,
                ),
            ]
        )

        metrics = reporter.condition_metrics(
            ExperimentCondition.POISONED_DEFENSE
        )

        assert metrics.vulnerability_rate == pytest.approx(
            0.5
        )

    def test_condition_metadata_contains_experiment_ids(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            [
                make_result(
                    experiment_id="exp-a",
                    condition=(
                        ExperimentCondition.CLEAN_NO_DEFENSE
                    ),
                ),
                make_result(
                    experiment_id="exp-b",
                    condition=(
                        ExperimentCondition.CLEAN_NO_DEFENSE
                    ),
                ),
            ]
        )

        metrics = reporter.condition_metrics(
            ExperimentCondition.CLEAN_NO_DEFENSE
        )

        assert metrics.metadata["experiment_ids"] == [
            "exp-a",
            "exp-b",
        ]

    def test_poisoned_defense_comparison_can_show_negative_utility_delta(
        self,
    ) -> None:
        reporter = ExperimentReporter(
            make_all_conditions()
        )

        comparison = reporter.compare_conditions(
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
        )

        assert comparison.utility_score_delta < 0.0
        assert comparison.attack_success_rate_delta < 0.0