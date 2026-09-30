from __future__ import annotations

import pytest

from src.evaluation.security_utility import (
    ConfusionMatrix,
    SecurityUtilityInputError,
    SecurityUtilityObservation,
    calculate_confusion_matrix,
    calculate_generation_quality_change,
    calculate_retrieval_quality_change,
    calculate_security_improvement,
    calculate_security_utility_report,
    calculate_security_utility_score,
    mean_generation_quality,
    mean_retrieval_quality,
    mean_security_score,
)


def make_observation(
    experiment_id: str = "exp-1",
    condition: str = "CLEAN_NO_DEFENSE",
    security_score: float = 0.9,
    retrieval_quality: float = 0.9,
    generation_quality: float = 0.9,
    defense_enabled: bool = False,
    blocked: bool = False,
    attack_succeeded: bool = False,
    vulnerability_introduced: bool = False,
    latency_ms: float | None = 10.0,
) -> SecurityUtilityObservation:
    return SecurityUtilityObservation(
        experiment_id=experiment_id,
        condition=condition,
        security_score=security_score,
        retrieval_quality=retrieval_quality,
        generation_quality=generation_quality,
        defense_enabled=defense_enabled,
        blocked=blocked,
        attack_succeeded=attack_succeeded,
        vulnerability_introduced=vulnerability_introduced,
        latency_ms=latency_ms,
    )


class TestSecurityUtilityObservation:
    def test_valid_observation(self) -> None:
        observation = make_observation()

        assert observation.experiment_id == "exp-1"
        assert observation.condition == "CLEAN_NO_DEFENSE"

    def test_serialization(self) -> None:
        observation = make_observation()

        data = observation.as_dict()

        assert data["experiment_id"] == "exp-1"
        assert data["security_score"] == 0.9

    def test_empty_experiment_id_is_rejected(self) -> None:
        with pytest.raises(SecurityUtilityInputError):
            make_observation(experiment_id="")

    def test_empty_condition_is_rejected(self) -> None:
        with pytest.raises(SecurityUtilityInputError):
            make_observation(condition="")

    def test_security_score_must_be_normalized(self) -> None:
        with pytest.raises(SecurityUtilityInputError):
            make_observation(security_score=1.1)

    def test_negative_latency_is_rejected(self) -> None:
        with pytest.raises(SecurityUtilityInputError):
            make_observation(latency_ms=-1.0)


class TestConfusionMatrix:
    def test_empty_matrix(self) -> None:
        matrix = ConfusionMatrix(
            true_positive=0,
            true_negative=0,
            false_positive=0,
            false_negative=0,
        )

        assert matrix.total == 0
        assert matrix.accuracy == 0.0
        assert matrix.precision == 0.0
        assert matrix.recall == 0.0
        assert matrix.f1 == 0.0

    def test_perfect_classification(self) -> None:
        matrix = ConfusionMatrix(
            true_positive=5,
            true_negative=5,
            false_positive=0,
            false_negative=0,
        )

        assert matrix.accuracy == 1.0
        assert matrix.precision == 1.0
        assert matrix.recall == 1.0
        assert matrix.f1 == 1.0

    def test_serialization(self) -> None:
        matrix = ConfusionMatrix(
            true_positive=2,
            true_negative=3,
            false_positive=1,
            false_negative=4,
        )

        data = matrix.as_dict()

        assert data["total"] == 10
        assert data["true_positive"] == 2


class TestConfusionMatrixCalculation:
    def test_calculates_tp_tn_fp_fn(self) -> None:
        observations = [
            make_observation(
                experiment_id="tp",
                attack_succeeded=True,
                blocked=True,
            ),
            make_observation(
                experiment_id="tn",
                attack_succeeded=False,
                blocked=False,
            ),
            make_observation(
                experiment_id="fp",
                attack_succeeded=False,
                blocked=True,
            ),
            make_observation(
                experiment_id="fn",
                attack_succeeded=True,
                blocked=False,
            ),
        ]

        matrix = calculate_confusion_matrix(observations)

        assert matrix.true_positive == 1
        assert matrix.true_negative == 1
        assert matrix.false_positive == 1
        assert matrix.false_negative == 1


class TestMeanMetrics:
    def test_mean_security(self) -> None:
        observations = [
            make_observation(
                experiment_id="1",
                security_score=0.8,
            ),
            make_observation(
                experiment_id="2",
                security_score=1.0,
            ),
        ]

        assert mean_security_score(observations) == pytest.approx(
            0.9
        )

    def test_mean_retrieval(self) -> None:
        observations = [
            make_observation(
                experiment_id="1",
                retrieval_quality=0.8,
            ),
            make_observation(
                experiment_id="2",
                retrieval_quality=1.0,
            ),
        ]

        assert mean_retrieval_quality(
            observations
        ) == pytest.approx(0.9)

    def test_mean_generation(self) -> None:
        observations = [
            make_observation(
                experiment_id="1",
                generation_quality=0.6,
            ),
            make_observation(
                experiment_id="2",
                generation_quality=0.8,
            ),
        ]

        assert mean_generation_quality(
            observations
        ) == pytest.approx(0.7)

    def test_empty_observations_return_zero(self) -> None:
        assert mean_security_score([]) == 0.0
        assert mean_retrieval_quality([]) == 0.0
        assert mean_generation_quality([]) == 0.0


class TestBaselineChanges:
    def test_security_improvement(self) -> None:
        observations = [
            make_observation(
                experiment_id="baseline",
                condition="CLEAN_NO_DEFENSE",
                security_score=0.5,
            ),
            make_observation(
                experiment_id="defense",
                condition="POISONED_DEFENSE",
                security_score=0.9,
                defense_enabled=True,
            ),
        ]

        assert calculate_security_improvement(
            observations
        ) == pytest.approx(0.2)

    def test_retrieval_quality_change(self) -> None:
        observations = [
            make_observation(
                experiment_id="baseline",
                condition="CLEAN_NO_DEFENSE",
                retrieval_quality=0.9,
            ),
            make_observation(
                experiment_id="defense",
                condition="POISONED_DEFENSE",
                retrieval_quality=0.8,
                defense_enabled=True,
            ),
        ]

        assert calculate_retrieval_quality_change(
            observations
        ) == pytest.approx(-0.1)

    def test_generation_quality_change(self) -> None:
        observations = [
            make_observation(
                experiment_id="baseline",
                condition="CLEAN_NO_DEFENSE",
                generation_quality=0.9,
            ),
            make_observation(
                experiment_id="defense",
                condition="POISONED_DEFENSE",
                generation_quality=0.85,
                defense_enabled=True,
            ),
        ]

        assert calculate_generation_quality_change(
            observations
        ) == pytest.approx(-0.05)


class TestSecurityUtilityScore:
    def test_combined_score(self) -> None:
        score = calculate_security_utility_score(
            security_score=1.0,
            retrieval_quality=0.8,
            generation_quality=0.6,
        )

        assert score == pytest.approx(0.85)

    def test_score_is_normalized(self) -> None:
        score = calculate_security_utility_score(
            security_score=1.0,
            retrieval_quality=1.0,
            generation_quality=1.0,
        )

        assert score == 1.0

    def test_invalid_score_is_rejected(self) -> None:
        with pytest.raises(SecurityUtilityInputError):
            calculate_security_utility_score(
                security_score=2.0,
                retrieval_quality=0.5,
                generation_quality=0.5,
            )


class TestSecurityUtilityReport:
    def test_report_is_created(self) -> None:
        observations = [
            make_observation(
                experiment_id="baseline",
                condition="CLEAN_NO_DEFENSE",
                security_score=0.5,
                retrieval_quality=0.9,
                generation_quality=0.9,
            ),
            make_observation(
                experiment_id="defense",
                condition="POISONED_DEFENSE",
                security_score=0.9,
                retrieval_quality=0.8,
                generation_quality=0.85,
                defense_enabled=True,
                blocked=True,
                attack_succeeded=True,
                latency_ms=20.0,
            ),
        ]

        report = calculate_security_utility_report(
            observations,
            metadata={"phase": 17},
        )

        assert report.observation_count == 2
        assert report.mean_security_score == pytest.approx(
            0.7
        )
        assert report.mean_retrieval_quality == pytest.approx(
            0.85
        )
        assert report.mean_generation_quality == pytest.approx(
            0.875
        )
        assert report.security_improvement == pytest.approx(
            0.2
        )
        assert report.retrieval_quality_change == pytest.approx(
            -0.05
        )
        assert report.generation_quality_change == pytest.approx(
            -0.025
        )
        assert report.mean_latency_ms == pytest.approx(
            15.0
        )
        assert report.confusion_matrix.true_positive == 1
        assert report.metadata["phase"] == 17

    def test_empty_report(self) -> None:
        report = calculate_security_utility_report([])

        assert report.observation_count == 0
        assert report.mean_security_score == 0.0
        assert report.mean_retrieval_quality == 0.0
        assert report.mean_generation_quality == 0.0
        assert report.mean_latency_ms is None

    def test_report_serialization(self) -> None:
        report = calculate_security_utility_report(
            [make_observation()]
        )

        data = report.as_dict()

        assert "security_improvement" in data
        assert "retrieval_quality_change" in data
        assert "generation_quality_change" in data
        assert "confusion_matrix" in data