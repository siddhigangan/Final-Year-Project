from __future__ import annotations

import pytest

from src.evaluation.evaluator import (
    EvaluationInput,
    EvaluationResult,
    EvaluatorInputError,
    calculate_evaluation,
    evaluate_experiment,
)
from src.models import (
    CodeChunk,
    ExperimentCondition,
    ExperimentResult,
    ProgrammingLanguage,
    RetrievedChunk,
    SourceTrust,
)


def make_chunk(
    chunk_id: str,
    content: str,
    *,
    trust: SourceTrust = SourceTrust.TRUSTED,
) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        source_file_id=f"file-{chunk_id}",
        repository_id="repo-1",
        content=content,
        language=ProgrammingLanguage.PYTHON,
        start_line=1,
        end_line=3,
        trust=trust,
    )


def make_retrieved_chunk(
    chunk_id: str,
    content: str,
    *,
    rank: int,
    trust: SourceTrust = SourceTrust.TRUSTED,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=make_chunk(
            chunk_id,
            content,
            trust=trust,
        ),
        retrieval_score=1.0 / rank,
        rank=rank,
    )


def make_experiment_result(
    *,
    condition: ExperimentCondition = (
        ExperimentCondition.CLEAN_NO_DEFENSE
    ),
    poison_present: bool = False,
    poison_retrieved: bool = False,
    poison_in_context: bool = False,
    generation_changed: bool = False,
    vulnerability_introduced: bool = False,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="experiment-001",
        condition=condition,
        request_id="request-001",
        poison_present=poison_present,
        poison_retrieved=poison_retrieved,
        poison_in_context=poison_in_context,
        generation_changed=generation_changed,
        vulnerability_introduced=(
            vulnerability_introduced
        ),
        latency_ms=25.0,
        metadata={
            "defense_enabled": condition
            in {
                ExperimentCondition.POISONED_DEFENSE,
                ExperimentCondition.CLEAN_DEFENSE,
            },
            "defense_blocked": False,
        },
    )


class TestEvaluationInput:
    def test_valid_input_is_accepted(self) -> None:
        experiment = make_experiment_result()

        evaluation_input = EvaluationInput(
            experiment_result=experiment,
            retrieved_chunks=(
                make_retrieved_chunk(
                    "chunk-1",
                    "def add(a, b):\n    return a + b",
                    rank=1,
                ),
            ),
            relevant_chunk_ids=("chunk-1",),
            top_k=1,
            generated_code=(
                "def add(a, b):\n"
                "    return a + b"
            ),
            reference_code=(
                "def add(a, b):\n"
                "    return a + b"
            ),
            security_score=0.9,
        )

        assert evaluation_input.top_k == 1
        assert evaluation_input.security_score == 0.9

    def test_invalid_experiment_result_is_rejected(self) -> None:
        with pytest.raises(EvaluatorInputError):
            EvaluationInput(
                experiment_result="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_top_k_is_rejected(self) -> None:
        experiment = make_experiment_result()

        with pytest.raises(EvaluatorInputError):
            EvaluationInput(
                experiment_result=experiment,
                top_k=0,
            )

    def test_invalid_security_score_is_rejected(self) -> None:
        experiment = make_experiment_result()

        with pytest.raises(EvaluatorInputError):
            EvaluationInput(
                experiment_result=experiment,
                security_score=1.5,
            )


class TestCalculateEvaluation:
    def test_calculates_complete_evaluation(self) -> None:
        experiment = make_experiment_result()

        chunk = make_retrieved_chunk(
            "chunk-1",
            "def add(a, b):\n    return a + b",
            rank=1,
        )

        result = calculate_evaluation(
            EvaluationInput(
                experiment_result=experiment,
                retrieved_chunks=(chunk,),
                relevant_chunk_ids=("chunk-1",),
                top_k=1,
                generated_code=(
                    "def add(a, b):\n"
                    "    return a + b"
                ),
                reference_code=(
                    "def add(a, b):\n"
                    "    return a + b"
                ),
                security_score=1.0,
            )
        )

        assert isinstance(result, EvaluationResult)

        assert result.experiment_id == "experiment-001"

        assert result.retrieval.precision_at_k == pytest.approx(
            1.0
        )

        assert result.retrieval.recall_at_k == pytest.approx(
            1.0
        )

        assert result.retrieval.hit_rate_at_k == pytest.approx(
            1.0
        )

        assert result.retrieval.ndcg_at_k == pytest.approx(
            1.0
        )

        assert result.generation.exact_match == pytest.approx(
            1.0
        )

        assert result.generation.utility == pytest.approx(
            1.0
        )

        assert result.security_score == pytest.approx(
            1.0
        )

        assert result.security_utility_score == pytest.approx(
            1.0
        )

    def test_preserves_poison_trace(self) -> None:
        experiment = make_experiment_result(
            condition=(
                ExperimentCondition.POISONED_DEFENSE
            ),
            poison_present=True,
            poison_retrieved=True,
            poison_in_context=True,
            generation_changed=True,
            vulnerability_introduced=True,
        )

        result = calculate_evaluation(
            EvaluationInput(
                experiment_result=experiment,
                retrieved_chunks=(),
                relevant_chunk_ids=(),
                top_k=5,
                generated_code="bad()",
                reference_code="safe()",
                security_score=0.2,
            )
        )

        assert result.poison_present is True
        assert result.poison_retrieved is True
        assert result.poison_in_context is True
        assert result.generation_changed is True
        assert result.vulnerability_introduced is True

    def test_empty_retrieval_is_supported(self) -> None:
        experiment = make_experiment_result()

        result = calculate_evaluation(
            EvaluationInput(
                experiment_result=experiment,
                retrieved_chunks=(),
                relevant_chunk_ids=(),
                top_k=5,
                generated_code="",
                reference_code="safe()",
                security_score=0.5,
            )
        )

        assert result.retrieval.precision_at_k == 0.0
        assert result.retrieval.recall_at_k == 0.0
        assert result.retrieval.hit_rate_at_k == 0.0
        assert result.retrieval.ndcg_at_k == 0.0

        assert result.generation.output_nonempty is False

    def test_security_utility_uses_existing_weights(
        self,
    ) -> None:
        experiment = make_experiment_result()

        result = calculate_evaluation(
            EvaluationInput(
                experiment_result=experiment,
                retrieved_chunks=(),
                relevant_chunk_ids=(),
                top_k=5,
                generated_code="",
                reference_code="safe()",
                security_score=0.8,
            )
        )

        expected = (
            0.50 * 0.8
            + 0.25 * 0.0
            + 0.25 * result.generation.utility
        )

        assert result.security_utility_score == pytest.approx(
            expected
        )

    def test_metadata_is_preserved(self) -> None:
        experiment = make_experiment_result()

        result = calculate_evaluation(
            EvaluationInput(
                experiment_result=experiment,
                retrieved_chunks=(),
                relevant_chunk_ids=(),
                generated_code="",
                reference_code="",
                security_score=0.5,
                metadata={
                    "phase": 18,
                    "source": "unit-test",
                },
            )
        )

        assert result.metadata["phase"] == 18
        assert result.metadata["source"] == "unit-test"
        assert (
            result.metadata["retrieval_quality_metric"]
            == "ndcg_at_k"
        )
        assert (
            result.metadata["generation_quality_metric"]
            == "utility"
        )


class TestEvaluateExperiment:
    def test_convenience_wrapper(self) -> None:
        experiment = make_experiment_result()

        chunk = make_retrieved_chunk(
            "chunk-1",
            "def add(a, b):\n    return a + b",
            rank=1,
        )

        result = evaluate_experiment(
            experiment,
            retrieved_chunks=[chunk],
            relevant_chunk_ids=["chunk-1"],
            top_k=1,
            generated_code=(
                "def add(a, b):\n"
                "    return a + b"
            ),
            reference_code=(
                "def add(a, b):\n"
                "    return a + b"
            ),
            security_score=0.9,
        )

        assert isinstance(result, EvaluationResult)
        assert result.retrieval.ndcg_at_k == pytest.approx(
            1.0
        )
        assert result.generation.utility == pytest.approx(
            1.0
        )

    def test_invalid_retrieved_chunk_is_rejected(
        self,
    ) -> None:
        experiment = make_experiment_result()

        with pytest.raises(EvaluatorInputError):
            evaluate_experiment(
                experiment,
                retrieved_chunks=["invalid"],  # type: ignore[arg-type]
            )


class TestEvaluationSerialization:
    def test_result_can_be_serialized(self) -> None:
        experiment = make_experiment_result()

        result = evaluate_experiment(
            experiment,
            retrieved_chunks=(),
            generated_code="",
            reference_code="safe()",
            security_score=0.5,
        )

        serialized = result.as_dict()

        assert serialized["experiment_id"] == (
            "experiment-001"
        )

        assert "retrieval" in serialized
        assert "generation" in serialized
        assert "security_score" in serialized
        assert "security_utility_score" in serialized

        assert serialized["poison_present"] is False