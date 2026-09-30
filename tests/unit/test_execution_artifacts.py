"""Unit tests for execution artifacts."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.evaluation.execution_artifacts import (
    ExecutionArtifacts,
    ExecutionArtifactsInputError,
    build_execution_artifacts,
)
from src.models import (
    ExperimentCondition,
    ExperimentResult,
)


def make_result(
    *,
    experiment_id: str = "exp-001",
    condition: ExperimentCondition = (
        ExperimentCondition.POISONED_DEFENSE
    ),
) -> ExperimentResult:
    """Create a valid experiment result for testing."""

    return ExperimentResult(
        experiment_id=experiment_id,
        condition=condition,
        request_id="request-001",
        poison_present=True,
        poison_retrieved=True,
        poison_in_context=True,
        generation_changed=False,
        vulnerability_introduced=False,
        poison_retrieval_rate=0.5,
        attack_success_rate=0.2,
        vulnerability_introduction_rate=0.1,
        utility_score=0.85,
        latency_ms=25.0,
        created_at=datetime.now(timezone.utc),
    )


def test_valid_execution_artifacts_are_created() -> None:
    """A valid artifact object should be created successfully."""

    result = make_result()

    artifacts = ExecutionArtifacts(
        experiment_id="exp-001",
        condition=ExperimentCondition.POISONED_DEFENSE,
        query="Implement a login function.",
        retrieved_chunks=(),
        relevant_chunk_ids=(),
        generated_code="def login():\n    pass",
        reference_code="def login():\n    pass",
        defense_enabled=True,
        experiment_result=result,
    )

    assert artifacts.experiment_id == "exp-001"
    assert artifacts.condition == ExperimentCondition.POISONED_DEFENSE
    assert artifacts.defense_enabled is True
    assert artifacts.retrieved_count == 0
    assert artifacts.relevant_count == 0
    assert artifacts.output_nonempty is True


def test_empty_generated_code_is_allowed() -> None:
    """Empty generated output can be recorded for evaluation."""

    result = make_result()

    artifacts = ExecutionArtifacts(
        experiment_id="exp-001",
        condition=ExperimentCondition.POISONED_DEFENSE,
        query="Generate code.",
        retrieved_chunks=(),
        relevant_chunk_ids=(),
        generated_code="",
        reference_code="def test():\n    pass",
        defense_enabled=True,
        experiment_result=result,
    )

    assert artifacts.output_nonempty is False


def test_empty_experiment_id_is_rejected() -> None:
    """An empty experiment ID must be rejected."""

    result = make_result()

    with pytest.raises(ExecutionArtifactsInputError):
        ExecutionArtifacts(
            experiment_id="",
            condition=ExperimentCondition.POISONED_DEFENSE,
            query="Generate code.",
            retrieved_chunks=(),
            relevant_chunk_ids=(),
            generated_code="code",
            reference_code="code",
            defense_enabled=True,
            experiment_result=result,
        )


def test_invalid_condition_is_rejected() -> None:
    """An invalid condition must be rejected."""

    result = make_result()

    with pytest.raises(ExecutionArtifactsInputError):
        ExecutionArtifacts(
            experiment_id="exp-001",
            condition="invalid",  # type: ignore[arg-type]
            query="Generate code.",
            retrieved_chunks=(),
            relevant_chunk_ids=(),
            generated_code="code",
            reference_code="code",
            defense_enabled=True,
            experiment_result=result,
        )


def test_retrieved_chunks_must_be_tuple() -> None:
    """Retrieved chunks must use the immutable tuple representation."""

    result = make_result()

    with pytest.raises(ExecutionArtifactsInputError):
        ExecutionArtifacts(
            experiment_id="exp-001",
            condition=ExperimentCondition.POISONED_DEFENSE,
            query="Generate code.",
            retrieved_chunks=[],  # type: ignore[arg-type]
            relevant_chunk_ids=(),
            generated_code="code",
            reference_code="code",
            defense_enabled=True,
            experiment_result=result,
        )


def test_relevant_chunk_ids_must_be_valid() -> None:
    """Relevant chunk identifiers must be non-empty strings."""

    result = make_result()

    with pytest.raises(ExecutionArtifactsInputError):
        ExecutionArtifacts(
            experiment_id="exp-001",
            condition=ExperimentCondition.POISONED_DEFENSE,
            query="Generate code.",
            retrieved_chunks=(),
            relevant_chunk_ids=("",),  # type: ignore[arg-type]
            generated_code="code",
            reference_code="code",
            defense_enabled=True,
            experiment_result=result,
        )


def test_defense_enabled_must_be_boolean() -> None:
    """Defense enabled must be a real boolean."""

    result = make_result()

    with pytest.raises(ExecutionArtifactsInputError):
        ExecutionArtifacts(
            experiment_id="exp-001",
            condition=ExperimentCondition.POISONED_DEFENSE,
            query="Generate code.",
            retrieved_chunks=(),
            relevant_chunk_ids=(),
            generated_code="code",
            reference_code="code",
            defense_enabled=1,  # type: ignore[arg-type]
            experiment_result=result,
        )


def test_experiment_result_must_match_experiment_id() -> None:
    """The experiment result must belong to the same experiment."""

    result = make_result(experiment_id="different-exp")

    with pytest.raises(ExecutionArtifactsInputError):
        ExecutionArtifacts(
            experiment_id="exp-001",
            condition=ExperimentCondition.POISONED_DEFENSE,
            query="Generate code.",
            retrieved_chunks=(),
            relevant_chunk_ids=(),
            generated_code="code",
            reference_code="code",
            defense_enabled=True,
            experiment_result=result,
        )


def test_experiment_result_must_match_condition() -> None:
    """The experiment result must use the same condition."""

    result = make_result(
        condition=ExperimentCondition.CLEAN_NO_DEFENSE
    )

    with pytest.raises(ExecutionArtifactsInputError):
        ExecutionArtifacts(
            experiment_id="exp-001",
            condition=ExperimentCondition.POISONED_DEFENSE,
            query="Generate code.",
            retrieved_chunks=(),
            relevant_chunk_ids=(),
            generated_code="code",
            reference_code="code",
            defense_enabled=True,
            experiment_result=result,
        )


def test_metadata_is_preserved() -> None:
    """Metadata should survive artifact construction."""

    result = make_result()

    artifacts = ExecutionArtifacts(
        experiment_id="exp-001",
        condition=ExperimentCondition.POISONED_DEFENSE,
        query="Generate code.",
        retrieved_chunks=(),
        relevant_chunk_ids=(),
        generated_code="code",
        reference_code="reference",
        defense_enabled=True,
        experiment_result=result,
        metadata={"phase": 18, "source": "unit-test"},
    )

    assert artifacts.metadata["phase"] == 18
    assert artifacts.metadata["source"] == "unit-test"


def test_as_dict_is_serializable() -> None:
    """The artifact representation should be dictionary-based."""

    result = make_result()

    artifacts = ExecutionArtifacts(
        experiment_id="exp-001",
        condition=ExperimentCondition.POISONED_DEFENSE,
        query="Generate code.",
        retrieved_chunks=(),
        relevant_chunk_ids=("chunk-001",),
        generated_code="code",
        reference_code="reference",
        defense_enabled=True,
        experiment_result=result,
        metadata={"phase": 18},
    )

    data = artifacts.as_dict()

    assert data["experiment_id"] == "exp-001"
    assert data["condition"] == "C_poisoned_defense"
    assert data["defense_enabled"] is True
    assert data["relevant_chunk_ids"] == ["chunk-001"]
    assert data["retrieved_count"] == 0
    assert data["relevant_count"] == 1
    assert data["output_nonempty"] is True


def test_builder_creates_valid_artifacts() -> None:
    """The convenience builder should create valid artifacts."""

    result = make_result()

    artifacts = build_execution_artifacts(
        experiment_id="exp-001",
        condition=ExperimentCondition.POISONED_DEFENSE,
        query="Generate code.",
        retrieved_chunks=(),
        relevant_chunk_ids=(),
        generated_code="code",
        reference_code="reference",
        defense_enabled=True,
        experiment_result=result,
        metadata={"phase": 18},
    )

    assert isinstance(artifacts, ExecutionArtifacts)
    assert artifacts.defense_enabled is True
    assert artifacts.metadata["phase"] == 18