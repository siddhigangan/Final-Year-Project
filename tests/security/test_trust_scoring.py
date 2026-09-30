from __future__ import annotations

import pytest

from src.models import (
    CodeChunk,
    ProgrammingLanguage,
    RetrievedChunk,
    SourceTrust,
)
from src.security.trust_scoring import (
    SourceTrustScorer,
    TrustAssessment,
    TrustBatchAssessment,
    TrustScoringConfig,
    TrustScoringInputError,
)


def make_chunk(
    *,
    chunk_id: str = "chunk-1",
    source_file_id: str = "file-1",
    repository_id: str = "repo-1",
    trust: SourceTrust = SourceTrust.TRUSTED,
    is_documentation: bool = False,
    metadata: dict | None = None,
) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        source_file_id=source_file_id,
        repository_id=repository_id,
        content="def hello():\n    return 'hello'",
        language=ProgrammingLanguage.PYTHON,
        start_line=1,
        end_line=2,
        symbol_name="hello",
        symbol_type="function",
        ast_node_type="function_definition",
        parent_symbol=None,
        is_documentation=is_documentation,
        trust=trust,
        metadata=metadata or {},
    )


def make_retrieved_chunk(
    *,
    chunk_id: str = "chunk-1",
    trust: SourceTrust = SourceTrust.TRUSTED,
    is_documentation: bool = False,
    metadata: dict | None = None,
    retrieval_score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=make_chunk(
            chunk_id=chunk_id,
            trust=trust,
            is_documentation=is_documentation,
            metadata=metadata,
        ),
        retrieval_score=retrieval_score,
        rank=1,
        retriever_name="test-retriever",
    )


class TestTrustScoringConfig:
    def test_default_values(self):
        config = TrustScoringConfig()

        assert config.trusted_score == pytest.approx(1.0)
        assert config.unknown_score == pytest.approx(0.60)
        assert config.suspicious_score == pytest.approx(0.25)
        assert config.poisoned_score == pytest.approx(0.0)
        assert config.documentation_penalty == pytest.approx(0.05)
        assert config.external_source_penalty == pytest.approx(0.10)
        assert config.suspicious_metadata_penalty == pytest.approx(0.20)

    def test_custom_values(self):
        config = TrustScoringConfig(
            trusted_score=0.95,
            unknown_score=0.55,
            suspicious_score=0.20,
            poisoned_score=0.0,
            documentation_penalty=0.10,
            external_source_penalty=0.15,
            suspicious_metadata_penalty=0.25,
        )

        assert config.trusted_score == pytest.approx(0.95)
        assert config.unknown_score == pytest.approx(0.55)
        assert config.suspicious_score == pytest.approx(0.20)
        assert config.documentation_penalty == pytest.approx(0.10)
        assert config.external_source_penalty == pytest.approx(0.15)
        assert config.suspicious_metadata_penalty == pytest.approx(0.25)


class TestSourceTrustScorer:
    def test_trusted_source_gets_full_score(self):
        scorer = SourceTrustScorer()

        assessment = scorer.score_chunk(
            make_chunk(trust=SourceTrust.TRUSTED)
        )

        assert isinstance(assessment, TrustAssessment)
        assert assessment.score == pytest.approx(1.0)
        assert assessment.trust == SourceTrust.TRUSTED
        assert assessment.is_trusted is True
        assert assessment.is_suspicious is False
        assert assessment.is_poisoned is False

    def test_unknown_source_gets_unknown_score(self):
        scorer = SourceTrustScorer()

        assessment = scorer.score_chunk(
            make_chunk(trust=SourceTrust.UNKNOWN)
        )

        assert assessment.score == pytest.approx(0.60)
        assert assessment.trust == SourceTrust.UNKNOWN
        assert assessment.is_trusted is False

    def test_suspicious_source_gets_low_score(self):
        scorer = SourceTrustScorer()

        assessment = scorer.score_chunk(
            make_chunk(trust=SourceTrust.SUSPICIOUS)
        )

        assert assessment.score == pytest.approx(0.25)
        assert assessment.trust == SourceTrust.SUSPICIOUS
        assert assessment.is_suspicious is True

    def test_poisoned_source_gets_zero_score(self):
        scorer = SourceTrustScorer()

        assessment = scorer.score_chunk(
            make_chunk(trust=SourceTrust.POISONED)
        )

        assert assessment.score == pytest.approx(0.0)
        assert assessment.trust == SourceTrust.POISONED
        assert assessment.is_poisoned is True

    def test_documentation_penalty(self):
        scorer = SourceTrustScorer()

        assessment = scorer.score_chunk(
            make_chunk(
                trust=SourceTrust.TRUSTED,
                is_documentation=True,
            )
        )

        assert assessment.score == pytest.approx(0.95)
        assert any(
            "documentation" in reason.lower()
            for reason in assessment.reasons
        )

    def test_suspicious_metadata_penalty(self):
        scorer = SourceTrustScorer()

        assessment = scorer.score_chunk(
            make_chunk(
                trust=SourceTrust.TRUSTED,
                metadata={"poisoned": True},
            )
        )

        assert assessment.score == pytest.approx(0.80)
        assert any(
            "suspicious" in reason.lower()
            for reason in assessment.reasons
        )

    def test_multiple_supported_penalties_are_combined(self):
        scorer = SourceTrustScorer()

        assessment = scorer.score_chunk(
            make_chunk(
                trust=SourceTrust.TRUSTED,
                is_documentation=True,
                metadata={"poisoned": True},
            )
        )

        assert assessment.score == pytest.approx(0.75)

    def test_false_suspicious_metadata_value_does_not_trigger(self):
        scorer = SourceTrustScorer()

        assessment = scorer.score_chunk(
            make_chunk(
                trust=SourceTrust.TRUSTED,
                metadata={"poisoned": False},
            )
        )

        assert assessment.score == pytest.approx(1.0)

    def test_score_never_goes_below_zero(self):
        config = TrustScoringConfig(
            suspicious_score=0.0,
            documentation_penalty=1.0,
            external_source_penalty=1.0,
            suspicious_metadata_penalty=1.0,
        )
        scorer = SourceTrustScorer(config)

        assessment = scorer.score_chunk(
            make_chunk(
                trust=SourceTrust.SUSPICIOUS,
                is_documentation=True,
                metadata={"poisoned": True},
            )
        )

        assert assessment.score >= 0.0

    def test_score_never_exceeds_one(self):
        config = TrustScoringConfig(
            trusted_score=2.0,
        )
        scorer = SourceTrustScorer(config)

        assessment = scorer.score_chunk(
            make_chunk(trust=SourceTrust.TRUSTED)
        )

        assert assessment.score <= 1.0

    def test_assessment_contains_source_identifiers(self):
        scorer = SourceTrustScorer()

        assessment = scorer.score_chunk(
            make_chunk(
                chunk_id="chunk-42",
                source_file_id="file-42",
                repository_id="repo-42",
            )
        )

        assert assessment.chunk_id == "chunk-42"
        assert assessment.source_file_id == "file-42"
        assert assessment.repository_id == "repo-42"

    def test_reasons_are_recorded(self):
        scorer = SourceTrustScorer()

        assessment = scorer.score_chunk(
            make_chunk(
                trust=SourceTrust.SUSPICIOUS,
                is_documentation=True,
            )
        )

        assert len(assessment.reasons) >= 2

    def test_retrieved_chunk_scoring(self):
        scorer = SourceTrustScorer()

        retrieved = make_retrieved_chunk(
            chunk_id="retrieved-1",
            trust=SourceTrust.TRUSTED,
        )

        assessment = scorer.score_retrieved_chunk(retrieved)

        assert assessment.chunk_id == "retrieved-1"
        assert assessment.score == pytest.approx(1.0)

    def test_batch_scoring_returns_batch_assessment(self):
        scorer = SourceTrustScorer()

        chunks = [
            make_chunk(
                chunk_id="trusted",
                trust=SourceTrust.TRUSTED,
            ),
            make_chunk(
                chunk_id="unknown",
                trust=SourceTrust.UNKNOWN,
            ),
            make_chunk(
                chunk_id="suspicious",
                trust=SourceTrust.SUSPICIOUS,
            ),
            make_chunk(
                chunk_id="poisoned",
                trust=SourceTrust.POISONED,
            ),
        ]

        batch = scorer.score_chunks(chunks)

        assert isinstance(batch, TrustBatchAssessment)
        assert batch.total_count == 4
        assert batch.trusted_count == 1
        assert batch.unknown_count == 1
        assert batch.suspicious_count == 1
        assert batch.poisoned_count == 1

    def test_batch_retrieved_chunk_scoring(self):
        scorer = SourceTrustScorer()

        chunks = [
            make_retrieved_chunk(
                chunk_id="one",
                trust=SourceTrust.TRUSTED,
            ),
            make_retrieved_chunk(
                chunk_id="two",
                trust=SourceTrust.SUSPICIOUS,
            ),
        ]

        batch = scorer.score_retrieved_chunks(chunks)

        assert isinstance(batch, TrustBatchAssessment)
        assert batch.total_count == 2
        assert batch.trusted_count == 1
        assert batch.suspicious_count == 1

    def test_batch_average_score(self):
        scorer = SourceTrustScorer()

        chunks = [
            make_chunk(
                chunk_id="one",
                trust=SourceTrust.TRUSTED,
            ),
            make_chunk(
                chunk_id="two",
                trust=SourceTrust.UNKNOWN,
            ),
        ]

        batch = scorer.score_chunks(chunks)

        assert batch.average_score == pytest.approx(0.80)
        assert batch.minimum_score == pytest.approx(0.60)
        assert batch.maximum_score == pytest.approx(1.0)

    def test_empty_batch(self):
        scorer = SourceTrustScorer()

        batch = scorer.score_chunks([])

        assert isinstance(batch, TrustBatchAssessment)
        assert batch.total_count == 0
        assert batch.average_score == pytest.approx(0.0)
        assert batch.minimum_score == pytest.approx(0.0)
        assert batch.maximum_score == pytest.approx(0.0)

    def test_custom_scoring_configuration(self):
        config = TrustScoringConfig(
            trusted_score=0.90,
            unknown_score=0.50,
            suspicious_score=0.20,
            poisoned_score=0.0,
            documentation_penalty=0.10,
        )

        scorer = SourceTrustScorer(config)

        assessment = scorer.score_chunk(
            make_chunk(
                trust=SourceTrust.TRUSTED,
                is_documentation=True,
            )
        )

        assert assessment.score == pytest.approx(0.80)

    def test_score_chunks_preserves_order_in_assessments(self):
        scorer = SourceTrustScorer()

        chunks = [
            make_chunk(
                chunk_id="first",
                trust=SourceTrust.TRUSTED,
            ),
            make_chunk(
                chunk_id="second",
                trust=SourceTrust.SUSPICIOUS,
            ),
            make_chunk(
                chunk_id="third",
                trust=SourceTrust.POISONED,
            ),
        ]

        batch = scorer.score_chunks(chunks)

        assert [item.chunk_id for item in batch.assessments] == [
            "first",
            "second",
            "third",
        ]

    def test_invalid_chunk_type(self):
        scorer = SourceTrustScorer()

        with pytest.raises(TrustScoringInputError):
            scorer.score_chunk("not-a-code-chunk")

    def test_invalid_retrieved_chunk_type(self):
        scorer = SourceTrustScorer()

        with pytest.raises(TrustScoringInputError):
            scorer.score_retrieved_chunk("not-a-retrieved-chunk")

    def test_invalid_batch_item(self):
        scorer = SourceTrustScorer()

        with pytest.raises(TrustScoringInputError):
            scorer.score_chunks(
                [
                    make_chunk(),
                    "invalid",
                ]
            )

    def test_invalid_retrieved_batch_item(self):
        scorer = SourceTrustScorer()

        with pytest.raises(TrustScoringInputError):
            scorer.score_retrieved_chunks(
                [
                    make_retrieved_chunk(),
                    "invalid",
                ]
            )

    def test_batch_assessment_properties(self):
        scorer = SourceTrustScorer()

        batch = scorer.score_chunks(
            [
                make_chunk(
                    trust=SourceTrust.TRUSTED,
                ),
                make_chunk(
                    chunk_id="suspicious",
                    trust=SourceTrust.SUSPICIOUS,
                ),
                make_chunk(
                    chunk_id="poisoned",
                    trust=SourceTrust.POISONED,
                ),
            ]
        )

        assert batch.total_count == 3
        assert batch.trusted_count == 1
        assert batch.suspicious_count == 1
        assert batch.poisoned_count == 1

    def test_assessment_properties(self):
        trusted = TrustAssessment(
            chunk_id="1",
            source_file_id="file-1",
            repository_id="repo-1",
            trust=SourceTrust.TRUSTED,
            score=1.0,
            reasons=(),
        )

        suspicious = TrustAssessment(
            chunk_id="2",
            source_file_id="file-2",
            repository_id="repo-1",
            trust=SourceTrust.SUSPICIOUS,
            score=0.25,
            reasons=(),
        )

        poisoned = TrustAssessment(
            chunk_id="3",
            source_file_id="file-3",
            repository_id="repo-1",
            trust=SourceTrust.POISONED,
            score=0.0,
            reasons=(),
        )

        assert trusted.is_trusted is True
        assert trusted.is_suspicious is False
        assert trusted.is_poisoned is False

        assert suspicious.is_trusted is False
        assert suspicious.is_suspicious is True
        assert suspicious.is_poisoned is False

        assert poisoned.is_trusted is False
        assert poisoned.is_poisoned is True

    def test_custom_suspicious_metadata_key(self):
        config = TrustScoringConfig(
            suspicious_metadata_keys=("dangerous",)
        )
        scorer = SourceTrustScorer(config)

        assessment = scorer.score_chunk(
            make_chunk(
                trust=SourceTrust.TRUSTED,
                metadata={"dangerous": True},
            )
        )

        assert assessment.score == pytest.approx(0.80)

    def test_metadata_can_contain_normal_values(self):
        scorer = SourceTrustScorer()

        assessment = scorer.score_chunk(
            make_chunk(
                trust=SourceTrust.TRUSTED,
                metadata={
                    "language": "python",
                    "symbol": "hello",
                    "line_count": 2,
                },
            )
        )

        assert assessment.score == pytest.approx(1.0)