from __future__ import annotations

import pytest

from src.evaluation.generation_metrics import (
    GenerationMetricsInputError,
    batch_generation_utility,
    calculate_generation_metrics,
    calculate_generation_utility,
    character_similarity,
    exact_match,
    normalize_code,
    normalized_exact_match,
    output_is_nonempty,
    token_f1,
    token_precision,
    token_recall,
)
from src.evaluation.retrieval_metrics import (
    RetrievalMetricsInputError,
    calculate_retrieval_metrics,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.models import (
    CodeChunk,
    ProgrammingLanguage,
    RetrievedChunk,
    SourceTrust,
)

# ---------------------------------------------------------------------------
# Retrieval test helpers
# ---------------------------------------------------------------------------


def make_retrieved_chunk(
    chunk_id: str,
    score: float = 1.0,
) -> RetrievedChunk:
    """Create a minimal valid RetrievedChunk for metric tests."""
    chunk = CodeChunk(
        chunk_id=chunk_id,
        source_file_id="test-file",
        repository_id="test-repository",
        content=f"def {chunk_id}():\n    return 1",
        language=ProgrammingLanguage.PYTHON,
        start_line=1,
        end_line=2,
        symbol_name=chunk_id,
        symbol_type="function",
        ast_node_type="function_definition",
        parent_symbol=None,
        is_documentation=False,
        trust=SourceTrust.TRUSTED,
        metadata={},
    )

    return RetrievedChunk(
        chunk=chunk,
        retrieval_score=score,
        rank=1,
        retriever_name="test",
    )


@pytest.fixture
def retrieved_chunks() -> list[RetrievedChunk]:
    return [
        make_retrieved_chunk("chunk-1", 0.99),
        make_retrieved_chunk("chunk-2", 0.90),
        make_retrieved_chunk("chunk-3", 0.80),
        make_retrieved_chunk("chunk-4", 0.70),
    ]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_perfect_precision(self, retrieved_chunks):
        assert precision_at_k(
            retrieved_chunks,
            {"chunk-1", "chunk-2", "chunk-3", "chunk-4"},
            4,
        ) == 1.0

    def test_half_precision(self, retrieved_chunks):
        assert precision_at_k(
            retrieved_chunks,
            {"chunk-1", "chunk-3"},
            4,
        ) == 0.5

    def test_zero_precision(self, retrieved_chunks):
        assert precision_at_k(
            retrieved_chunks,
            {"unknown"},
            3,
        ) == 0.0

    def test_k_smaller_than_results(self, retrieved_chunks):
        assert precision_at_k(
            retrieved_chunks,
            {"chunk-1"},
            1,
        ) == 1.0

    def test_invalid_k(self, retrieved_chunks):
        with pytest.raises(RetrievalMetricsInputError):
            precision_at_k(
                retrieved_chunks,
                {"chunk-1"},
                0,
            )


class TestRecallAtK:
    def test_full_recall(self, retrieved_chunks):
        assert recall_at_k(
            retrieved_chunks,
            {"chunk-1", "chunk-2"},
            2,
        ) == 1.0

    def test_partial_recall(self, retrieved_chunks):
        assert recall_at_k(
            retrieved_chunks,
            {"chunk-1", "chunk-2"},
            1,
        ) == 0.5

    def test_zero_recall(self, retrieved_chunks):
        assert recall_at_k(
            retrieved_chunks,
            {"unknown"},
            3,
        ) == 0.0

    def test_invalid_relevant_ids(self, retrieved_chunks):
        with pytest.raises(RetrievalMetricsInputError):
            recall_at_k(
                retrieved_chunks,
                None,
                2,
            )


class TestHitRateAtK:
    def test_hit_exists(self, retrieved_chunks):
        assert hit_rate_at_k(
            retrieved_chunks,
            {"chunk-2"},
            3,
        ) == 1.0

    def test_no_hit(self, retrieved_chunks):
        assert hit_rate_at_k(
            retrieved_chunks,
            {"unknown"},
            3,
        ) == 0.0

    def test_hit_outside_k(self, retrieved_chunks):
        assert hit_rate_at_k(
            retrieved_chunks,
            {"chunk-3"},
            2,
        ) == 0.0


class TestReciprocalRank:
    def test_first_result(self, retrieved_chunks):
        assert reciprocal_rank(
            retrieved_chunks,
            {"chunk-1"},
        ) == 1.0

    def test_second_result(self, retrieved_chunks):
        assert reciprocal_rank(
            retrieved_chunks,
            {"chunk-2"},
        ) == 0.5

    def test_third_result(self, retrieved_chunks):
        assert reciprocal_rank(
            retrieved_chunks,
            {"chunk-3"},
        ) == pytest.approx(1 / 3)

    def test_no_relevant_result(self, retrieved_chunks):
        assert reciprocal_rank(
            retrieved_chunks,
            {"unknown"},
        ) == 0.0


class TestNDCG:
    def test_perfect_ranking(self, retrieved_chunks):
        assert ndcg_at_k(
            retrieved_chunks,
            {"chunk-1", "chunk-2", "chunk-3"},
            3,
        ) == pytest.approx(1.0)

    def test_empty_relevant_set(self, retrieved_chunks):
        assert ndcg_at_k(
            retrieved_chunks,
            set(),
            3,
        ) == 0.0

    def test_single_result(self, retrieved_chunks):
        assert ndcg_at_k(
            retrieved_chunks,
            {"chunk-1"},
            1,
        ) == pytest.approx(1.0)

    def test_reversed_relevance_is_lower(self, retrieved_chunks):
        perfect = ndcg_at_k(
            retrieved_chunks,
            {"chunk-1", "chunk-2", "chunk-3"},
            3,
        )

        reversed_score = ndcg_at_k(
            retrieved_chunks,
            {"chunk-3"},
            3,
        )

        assert reversed_score < perfect


class TestCalculateRetrievalMetrics:
    def test_returns_expected_metrics(self, retrieved_chunks):
        result = calculate_retrieval_metrics(
            retrieved_chunks,
            {"chunk-1", "chunk-2"},
            4,
        )

        assert result.precision_at_k == 0.5
        assert result.recall_at_k == 1.0
        assert result.hit_rate_at_k == 1.0
        assert result.reciprocal_rank == 1.0
        assert result.k == 4
        assert result.retrieved_count == 4
        assert result.relevant_count == 2

    def test_invalid_relevance(self, retrieved_chunks):
        with pytest.raises(RetrievalMetricsInputError):
            calculate_retrieval_metrics(
                retrieved_chunks,
                None,
                4,
            )


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------


class TestNormalizeCode:
    def test_removes_extra_whitespace(self):
        code = "def hello():    \n    return 1   "

        result = normalize_code(code)

        assert result == "def hello():\n    return 1"

    def test_normalizes_empty_code(self):
        assert normalize_code("") == ""

    def test_removes_blank_lines(self):
        code = "def hello():\n\n\n    return 1"

        result = normalize_code(code)

        assert "\n\n" not in result


class TestExactMatch:
    def test_exact_match(self):
        assert exact_match(
            "return 1",
            "return 1",
        ) == 1.0

    def test_different_code(self):
        assert exact_match(
            "return 1",
            "return 2",
        ) == 0.0

    def test_empty_strings(self):
        assert exact_match("", "") == 1.0


class TestNormalizedExactMatch:
    def test_whitespace_does_not_change_result(self):
        reference = "def add(a, b):\n    return a + b"
        candidate = "def add(a, b):\n\n    return a + b"

        assert normalized_exact_match(
            candidate,
            reference,
        ) == 1.0

    def test_different_code(self):
        assert normalized_exact_match(
            "return a + b",
            "return a - b",
        ) == 0.0


class TestTokenMetrics:
    def test_perfect_precision(self):
        assert token_precision(
            "return a + b",
            "return a + b",
        ) == 1.0

    def test_perfect_recall(self):
        assert token_recall(
            "return a + b",
            "return a + b",
        ) == 1.0

    def test_perfect_f1(self):
        assert token_f1(
            "return a + b",
            "return a + b",
        ) == 1.0

    def test_different_code(self):
        assert token_f1(
            "return a + b",
            "return a - b",
        ) < 1.0

    def test_empty_candidate(self):
        assert token_precision(
            "",
            "return 1",
        ) == 0.0

    def test_empty_reference(self):
        assert token_recall(
            "return 1",
            "",
        ) == 0.0


class TestCharacterSimilarity:
    def test_identical_strings(self):
        assert character_similarity(
            "return 1",
            "return 1",
        ) == pytest.approx(1.0)

    def test_different_strings(self):
        result = character_similarity(
            "return 1",
            "return 2",
        )

        assert 0.0 < result < 1.0

    def test_empty_strings(self):
        assert character_similarity("", "") == 1.0


class TestOutputValidation:
    def test_nonempty_output(self):
        assert output_is_nonempty("return 1") is True

    def test_empty_output(self):
        assert output_is_nonempty("") is False

    def test_whitespace_output(self):
        assert output_is_nonempty("   ") is False


class TestGenerationUtility:
    def test_perfect_generation(self):
        result = calculate_generation_utility(
            "return a + b",
            "return a + b",
        )

        assert result == pytest.approx(1.0)

    def test_different_generation(self):
        result = calculate_generation_utility(
            "return a - b",
            "return a + b",
        )

        assert 0.0 <= result < 1.0

    def test_empty_generation(self):
        result = calculate_generation_utility(
            "",
            "return 1",
        )

        assert result == 0.0


class TestCalculateGenerationMetrics:
    def test_returns_metrics(self):
        result = calculate_generation_metrics(
            "return a + b",
            "return a + b",
        )

        assert result.exact_match == 1.0
        assert result.normalized_exact_match == 1.0
        assert result.token_precision == 1.0
        assert result.token_recall == 1.0
        assert result.token_f1 == 1.0
        assert result.character_similarity == pytest.approx(1.0)
        assert result.output_nonempty is True
        assert result.utility == pytest.approx(1.0)

    def test_different_generation(self):
        result = calculate_generation_metrics(
            "return a - b",
            "return a + b",
        )

        assert result.exact_match == 0.0
        assert result.normalized_exact_match == 0.0
        assert result.token_f1 < 1.0
        assert result.utility < 1.0


class TestBatchGenerationUtility:
    def test_batch_utility(self):
        result = batch_generation_utility(
            [
                "return 1",
                "return 2",
            ],
            [
                "return 1",
                "return 3",
            ],
        )

        assert 0.0 < result < 1.0

    def test_empty_batch(self):
        assert batch_generation_utility([], []) == 0.0

    def test_mismatched_batch_lengths(self):
        with pytest.raises(GenerationMetricsInputError):
            batch_generation_utility(
                ["return 1"],
                [],
            )