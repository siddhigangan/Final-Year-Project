from __future__ import annotations

import pytest

from src.models import (
    CodeChunk,
    ProgrammingLanguage,
    RetrievedChunk,
    SourceTrust,
)
from src.retrieval.context_builder import (
    BuiltContext,
    ContextSource,
)
from src.security.context_validation import (
    ContextValidationConfig,
    ContextValidationInputError,
    ContextValidator,
)


def make_chunk(
    *,
    chunk_id: str = "chunk-1",
    source_file_id: str = "file-1",
    repository_id: str = "repo-1",
    content: str = "def hello():\n    return 'hello'",
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
    trust: SourceTrust = SourceTrust.TRUSTED,
    metadata: dict | None = None,
) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        source_file_id=source_file_id,
        repository_id=repository_id,
        content=content,
        language=language,
        start_line=1,
        end_line=max(1, content.count("\n") + 1),
        symbol_name="hello",
        symbol_type="function",
        ast_node_type="function_definition",
        parent_symbol=None,
        is_documentation=False,
        trust=trust,
        metadata=metadata or {},
    )


def make_retrieved_chunk(
    *,
    chunk_id: str = "chunk-1",
    source_file_id: str = "file-1",
    repository_id: str = "repo-1",
    content: str = "def hello():\n    return 'hello'",
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
    trust: SourceTrust = SourceTrust.TRUSTED,
    retrieval_score: float = 0.9,
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=make_chunk(
            chunk_id=chunk_id,
            source_file_id=source_file_id,
            repository_id=repository_id,
            content=content,
            language=language,
            trust=trust,
        ),
        retrieval_score=retrieval_score,
        rank=rank,
        retriever_name="test-retriever",
    )


def make_context(
    *,
    text: str = "def hello():\n    return 'hello'",
    sources: tuple[ContextSource, ...] = (),
    query: str | None = "hello",
    truncated: bool = False,
) -> BuiltContext:
    return BuiltContext(
        text=text,
        sources=sources,
        query=query,
        total_characters=len(text),
        total_lines=text.count("\n") + 1 if text else 0,
        truncated=truncated,
        metadata={},
    )


def make_source(
    *,
    chunk_id: str = "chunk-1",
    source_file_id: str = "file-1",
    repository_id: str = "repo-1",
    rank: int = 1,
    source_index: int = 1,
    relative_path: str = "src/example.py",
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
    trust: SourceTrust = SourceTrust.TRUSTED,
    retrieval_score: float = 0.9,
    symbol_name: str = "hello",
    symbol_type: str = "function",
    is_documentation: bool = False,
    content: str = "def hello():\n    return 'hello'",
) -> ContextSource:
    return ContextSource(
        source_index=source_index,
        chunk_id=chunk_id,
        source_file_id=source_file_id,
        repository_id=repository_id,
        relative_path=relative_path,
        language=language,
        trust=trust,
        retrieval_score=retrieval_score,
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        is_documentation=is_documentation,
        content=content,
        rank=rank,
    )


class TestContextValidationConfig:
    def test_default_values(self):
        config = ContextValidationConfig()

        assert config.max_sources == 20
        assert config.max_context_characters == 50000
        assert config.max_duplicate_chunk_ratio == pytest.approx(0.50)
        assert config.require_repository_consistency is True
        assert config.require_language_consistency is False
        assert config.require_provenance is True

    def test_custom_values(self):
        config = ContextValidationConfig(
            max_sources=10,
            max_context_characters=1000,
            max_duplicate_chunk_ratio=0.25,
            require_repository_consistency=False,
            require_language_consistency=True,
            require_provenance=False,
        )

        assert config.max_sources == 10
        assert config.max_context_characters == 1000
        assert config.max_duplicate_chunk_ratio == pytest.approx(0.25)
        assert config.require_repository_consistency is False
        assert config.require_language_consistency is True
        assert config.require_provenance is False

    def test_invalid_max_sources(self):
        with pytest.raises(ValueError, match="max_sources"):
            ContextValidationConfig(max_sources=0)

    def test_invalid_context_character_limit(self):
        with pytest.raises(
            ValueError,
            match="max_context_characters",
        ):
            ContextValidationConfig(max_context_characters=0)

    def test_invalid_duplicate_ratio(self):
        with pytest.raises(
            ValueError,
            match="max_duplicate_chunk_ratio",
        ):
            ContextValidationConfig(max_duplicate_chunk_ratio=1.5)


class TestContextValidator:
    def test_valid_context(self):
        validator = ContextValidator()

        retrieved = [
            make_retrieved_chunk(),
        ]

        context = make_context(
            sources=(
                make_source(),
            ),
        )

        result = validator.validate(
            context,
            retrieved,
        )

        assert result.valid is True
        assert result.safe_to_use is True
        assert result.source_count == 1
        assert result.unique_source_count == 1
        assert result.duplicate_count == 0
        assert result.error_count == 0

    def test_source_count_is_recorded(self):
        validator = ContextValidator()

        retrieved = [
            make_retrieved_chunk(chunk_id="chunk-1"),
            make_retrieved_chunk(
                chunk_id="chunk-2",
                source_file_id="file-2",
            ),
        ]

        context = make_context(
            text="first\n\nsecond",
            sources=(
                make_source(chunk_id="chunk-1"),
                make_source(
                    chunk_id="chunk-2",
                    source_file_id="file-2",
                ),
            ),
        )

        result = validator.validate(
            context,
            retrieved,
        )

        assert result.source_count == 2
        assert result.unique_source_count == 2
        assert result.duplicate_count == 0

    def test_duplicate_sources_generate_warning_below_threshold(self):
        validator = ContextValidator()

        retrieved = [
            make_retrieved_chunk(chunk_id="chunk-1"),
            make_retrieved_chunk(chunk_id="chunk-1"),
            make_retrieved_chunk(chunk_id="chunk-2"),
        ]

        context = make_context(
            text="first\n\nfirst\n\nsecond",
        )

        result = validator.validate(
            context,
            retrieved,
        )

        duplicate_issues = [
            issue
            for issue in result.issues
            if issue.issue_type == "duplicate_sources"
        ]

        assert len(duplicate_issues) == 1
        assert duplicate_issues[0].severity == "warning"
        assert result.valid is True
        assert result.safe_to_use is True

    def test_excessive_duplicates_generate_error(self):
        config = ContextValidationConfig(
            max_duplicate_chunk_ratio=0.25,
        )
        validator = ContextValidator(config)

        retrieved = [
            make_retrieved_chunk(chunk_id="chunk-1"),
            make_retrieved_chunk(chunk_id="chunk-1"),
            make_retrieved_chunk(chunk_id="chunk-1"),
            make_retrieved_chunk(chunk_id="chunk-2"),
        ]

        context = make_context(
            text="first\n\nfirst\n\nfirst\n\nsecond",
        )

        result = validator.validate(
            context,
            retrieved,
        )

        duplicate_issues = [
            issue
            for issue in result.issues
            if issue.issue_type == "duplicate_sources"
        ]

        assert len(duplicate_issues) == 1
        assert duplicate_issues[0].severity == "error"
        assert result.valid is False
        assert result.safe_to_use is False

    def test_repository_inconsistency_is_error(self):
        validator = ContextValidator()

        retrieved = [
            make_retrieved_chunk(
                chunk_id="chunk-1",
                repository_id="repo-a",
            ),
            make_retrieved_chunk(
                chunk_id="chunk-2",
                source_file_id="file-2",
                repository_id="repo-b",
            ),
        ]

        context = make_context(
            text="repo a\n\nrepo b",
        )

        result = validator.validate(
            context,
            retrieved,
        )

        repository_issues = [
            issue
            for issue in result.issues
            if issue.issue_type == "repository_inconsistency"
        ]

        assert len(repository_issues) == 1
        assert repository_issues[0].severity == "error"
        assert result.valid is False
        assert result.safe_to_use is False

    def test_repository_consistency_can_be_disabled(self):
        config = ContextValidationConfig(
            require_repository_consistency=False,
        )
        validator = ContextValidator(config)

        retrieved = [
            make_retrieved_chunk(
                chunk_id="chunk-1",
                repository_id="repo-a",
            ),
            make_retrieved_chunk(
                chunk_id="chunk-2",
                source_file_id="file-2",
                repository_id="repo-b",
            ),
        ]

        result = validator.validate(
            make_context(text="a\n\nb"),
            retrieved,
        )

        assert "repository_inconsistency" not in {
            issue.issue_type
            for issue in result.issues
        }
        assert result.valid is True

    def test_language_inconsistency_is_disabled_by_default(self):
        validator = ContextValidator()

        retrieved = [
            make_retrieved_chunk(
                chunk_id="python",
                language=ProgrammingLanguage.PYTHON,
            ),
            make_retrieved_chunk(
                chunk_id="java",
                source_file_id="file-2",
                language=ProgrammingLanguage.JAVA,
            ),
        ]

        result = validator.validate(
            make_context(text="python\n\njava"),
            retrieved,
        )

        assert "language_inconsistency" not in {
            issue.issue_type
            for issue in result.issues
        }
        assert result.valid is True

    def test_language_inconsistency_can_be_enabled(self):
        config = ContextValidationConfig(
            require_language_consistency=True,
        )
        validator = ContextValidator(config)

        retrieved = [
            make_retrieved_chunk(
                chunk_id="python",
                language=ProgrammingLanguage.PYTHON,
            ),
            make_retrieved_chunk(
                chunk_id="java",
                source_file_id="file-2",
                language=ProgrammingLanguage.JAVA,
            ),
        ]

        result = validator.validate(
            make_context(text="python\n\njava"),
            retrieved,
        )

        language_issues = [
            issue
            for issue in result.issues
            if issue.issue_type == "language_inconsistency"
        ]

        assert len(language_issues) == 1
        assert language_issues[0].severity == "warning"
        assert result.valid is True

    def test_context_size_limit_is_enforced(self):
        config = ContextValidationConfig(
            max_context_characters=20,
        )
        validator = ContextValidator(config)

        context = make_context(
            text="x" * 21,
        )

        result = validator.validate(
            context,
            [make_retrieved_chunk()],
        )

        size_issues = [
            issue
            for issue in result.issues
            if issue.issue_type == "context_size_exceeded"
        ]

        assert len(size_issues) == 1
        assert size_issues[0].severity == "error"
        assert result.valid is False
        assert result.safe_to_use is False

    def test_context_within_size_limit_is_valid(self):
        config = ContextValidationConfig(
            max_context_characters=100,
        )
        validator = ContextValidator(config)

        result = validator.validate(
            make_context(text="short"),
            [make_retrieved_chunk()],
        )

        assert "context_size_exceeded" not in {
            issue.issue_type
            for issue in result.issues
        }

    def test_source_count_limit_is_enforced(self):
        config = ContextValidationConfig(
            max_sources=2,
        )
        validator = ContextValidator(config)

        retrieved = [
            make_retrieved_chunk(
                chunk_id="one",
            ),
            make_retrieved_chunk(
                chunk_id="two",
                source_file_id="file-2",
            ),
            make_retrieved_chunk(
                chunk_id="three",
                source_file_id="file-3",
            ),
        ]

        result = validator.validate(
            make_context(text="one\n\ntwo\n\nthree"),
            retrieved,
        )

        assert "source_count_exceeded" in {
            issue.issue_type
            for issue in result.issues
        }
        assert result.valid is False

    def test_context_provenance_mismatch_is_warning(self):
        validator = ContextValidator()

        retrieved = [
            make_retrieved_chunk(
                chunk_id="chunk-1",
            ),
            make_retrieved_chunk(
                chunk_id="chunk-2",
                source_file_id="file-2",
            ),
        ]

        context = make_context(
            text="first\n\nsecond",
            sources=(
                make_source(chunk_id="chunk-1"),
            ),
        )

        result = validator.validate(
            context,
            retrieved,
        )

        issues = [
            issue
            for issue in result.issues
            if issue.issue_type == "source_provenance_mismatch"
        ]

        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert result.valid is True

    def test_matching_context_provenance_has_no_warning(self):
        validator = ContextValidator()

        retrieved = [
            make_retrieved_chunk(
                chunk_id="chunk-1",
            ),
            make_retrieved_chunk(
                chunk_id="chunk-2",
                source_file_id="file-2",
            ),
        ]

        context = make_context(
            text="first\n\nsecond",
            sources=(
                make_source(chunk_id="chunk-1"),
                make_source(
                    chunk_id="chunk-2",
                    source_file_id="file-2",
                ),
            ),
        )

        result = validator.validate(
            context,
            retrieved,
        )

        assert "source_provenance_mismatch" not in {
            issue.issue_type
            for issue in result.issues
        }

    def test_validate_chunks_builds_temporary_context(self):
        validator = ContextValidator()

        retrieved = [
            make_retrieved_chunk(
                chunk_id="chunk-1",
                content="def one():\n    return 1",
            ),
            make_retrieved_chunk(
                chunk_id="chunk-2",
                source_file_id="file-2",
                content="def two():\n    return 2",
            ),
        ]

        result = validator.validate_chunks(retrieved)

        assert result.valid is True
        assert result.source_count == 2
        assert result.context_characters > 0

    def test_validate_chunks_detects_repository_mismatch(self):
        validator = ContextValidator()

        retrieved = [
            make_retrieved_chunk(
                chunk_id="chunk-1",
                repository_id="repo-a",
            ),
            make_retrieved_chunk(
                chunk_id="chunk-2",
                source_file_id="file-2",
                repository_id="repo-b",
            ),
        ]

        result = validator.validate_chunks(retrieved)

        assert result.valid is False
        assert "repository_inconsistency" in {
            issue.issue_type
            for issue in result.issues
        }

    def test_empty_context_and_sources_is_valid(self):
        validator = ContextValidator()

        result = validator.validate(
            make_context(
                text="",
                sources=(),
            ),
            [],
        )

        assert result.valid is True
        assert result.safe_to_use is True
        assert result.source_count == 0
        assert result.unique_source_count == 0
        assert result.duplicate_count == 0

    def test_repositories_are_recorded(self):
        validator = ContextValidator()

        retrieved = [
            make_retrieved_chunk(
                chunk_id="one",
                repository_id="repo-a",
            ),
        ]

        result = validator.validate(
            make_context(),
            retrieved,
        )

        assert result.repositories == ("repo-a",)

    def test_languages_are_recorded(self):
        validator = ContextValidator()

        retrieved = [
            make_retrieved_chunk(
                language=ProgrammingLanguage.PYTHON,
            ),
        ]

        result = validator.validate(
            make_context(),
            retrieved,
        )

        assert result.languages == ("python",)

    def test_context_metadata_is_preserved(self):
        validator = ContextValidator()

        context = make_context(
            query="find hello",
            truncated=True,
        )

        result = validator.validate(
            context,
            [make_retrieved_chunk()],
        )

        assert result.metadata["context_query"] == "find hello"
        assert result.metadata["truncated"] is True

    def test_missing_chunk_identifier_can_be_disabled(self):
        config = ContextValidationConfig(
            reject_missing_chunk_ids=False,
        )
        validator = ContextValidator(config)

        retrieved = [
            make_retrieved_chunk(),
        ]

        object.__setattr__(
            retrieved[0].chunk,
            "chunk_id",
            "",
        )

        result = validator.validate(
            make_context(),
            retrieved,
        )

        assert "missing_chunk_id" not in {
            issue.issue_type
            for issue in result.issues
        }

    def test_missing_provenance_checks_can_be_disabled(self):
        config = ContextValidationConfig(
            require_provenance=False,
        )
        validator = ContextValidator(config)

        retrieved = [
            make_retrieved_chunk(),
        ]

        object.__setattr__(
            retrieved[0].chunk,
            "chunk_id",
            "",
        )

        object.__setattr__(
            retrieved[0].chunk,
            "source_file_id",
            "",
        )

        object.__setattr__(
            retrieved[0].chunk,
            "repository_id",
            "",
        )

        result = validator.validate(
            make_context(),
            retrieved,
        )

        provenance_issue_types = {
            "missing_chunk_id",
            "missing_source_file_id",
            "missing_repository_id",
        }

        assert not (
            provenance_issue_types
            & {
                issue.issue_type
                for issue in result.issues
            }
        )

    def test_invalid_context_type(self):
        validator = ContextValidator()

        with pytest.raises(ContextValidationInputError):
            validator.validate(
                "invalid",
                [],
            )

    def test_invalid_retrieved_chunks_type(self):
        validator = ContextValidator()

        with pytest.raises(ContextValidationInputError):
            validator.validate(
                make_context(),
                "invalid",
            )

    def test_invalid_retrieved_chunk_item(self):
        validator = ContextValidator()

        with pytest.raises(ContextValidationInputError):
            validator.validate(
                make_context(),
                ["invalid"],
            )

    def test_invalid_validate_chunks_input(self):
        validator = ContextValidator()

        with pytest.raises(ContextValidationInputError):
            validator.validate_chunks("invalid")

    def test_issue_counts(self):
        config = ContextValidationConfig(
            max_context_characters=5,
        )
        validator = ContextValidator(config)

        result = validator.validate(
            make_context(text="123456"),
            [
                make_retrieved_chunk(
                    chunk_id="chunk-1",
                ),
                make_retrieved_chunk(
                    chunk_id="chunk-1",
                ),
            ],
        )

        assert result.issue_count >= 2
        assert result.error_count >= 1
        assert result.warning_count >= 0
        assert result.has_errors is True