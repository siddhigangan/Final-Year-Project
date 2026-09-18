"""Unit tests for SecureCodeRAG shared domain models."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.models import (
    CodeChunk,
    ExperimentCondition,
    ExperimentConfig,
    ExperimentResult,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    ModelValidationError,
    ProgrammingLanguage,
    RetrievedChunk,
    SecurityDecision,
    SecurityFinding,
    SecuritySeverity,
    SourceFile,
    SourceTrust,
)


class TestProgrammingLanguage(unittest.TestCase):
    """Tests for ProgrammingLanguage."""

    def test_supported_language_values(self) -> None:
        self.assertEqual(ProgrammingLanguage.PYTHON.value, "python")
        self.assertEqual(ProgrammingLanguage.JAVASCRIPT.value, "javascript")
        self.assertEqual(ProgrammingLanguage.JAVA.value, "java")
        self.assertEqual(ProgrammingLanguage.UNKNOWN.value, "unknown")


class TestSourceFile(unittest.TestCase):
    """Tests for SourceFile."""

    def test_valid_source_file(self) -> None:
        source_file = SourceFile(
            file_id="file-001",
            repository_id="repo-001",
            relative_path="src/example.py",
            language=ProgrammingLanguage.PYTHON,
            content="print('hello')",
            size_bytes=14,
            sha256="abc123",
        )

        self.assertEqual(source_file.file_id, "file-001")
        self.assertEqual(source_file.language, ProgrammingLanguage.PYTHON)
        self.assertEqual(source_file.trust, SourceTrust.TRUSTED)

    def test_empty_file_id_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            SourceFile(
                file_id="",
                repository_id="repo-001",
                relative_path="src/example.py",
                language=ProgrammingLanguage.PYTHON,
                content="print('hello')",
                size_bytes=14,
                sha256="abc123",
            )

    def test_negative_size_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            SourceFile(
                file_id="file-001",
                repository_id="repo-001",
                relative_path="src/example.py",
                language=ProgrammingLanguage.PYTHON,
                content="print('hello')",
                size_bytes=-1,
                sha256="abc123",
            )

    def test_source_file_serialization(self) -> None:
        source_file = SourceFile(
            file_id="file-001",
            repository_id="repo-001",
            relative_path="README.md",
            language=ProgrammingLanguage.UNKNOWN,
            content="# SecureCodeRAG",
            size_bytes=16,
            sha256="abc123",
            is_documentation=True,
        )

        data = source_file.to_dict()

        self.assertEqual(data["file_id"], "file-001")
        self.assertEqual(data["is_documentation"], True)
        self.assertEqual(data["language"], ProgrammingLanguage.UNKNOWN)


class TestCodeChunk(unittest.TestCase):
    """Tests for CodeChunk."""

    def create_chunk(self) -> CodeChunk:
        return CodeChunk(
            chunk_id="chunk-001",
            source_file_id="file-001",
            repository_id="repo-001",
            content="def add(a, b):\n    return a + b",
            language=ProgrammingLanguage.PYTHON,
            start_line=10,
            end_line=11,
            symbol_name="add",
            symbol_type="function",
            ast_node_type="function_definition",
        )

    def test_valid_chunk(self) -> None:
        chunk = self.create_chunk()

        self.assertEqual(chunk.chunk_id, "chunk-001")
        self.assertEqual(chunk.symbol_name, "add")
        self.assertEqual(chunk.line_count, 2)

    def test_empty_chunk_id_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            CodeChunk(
                chunk_id="",
                source_file_id="file-001",
                repository_id="repo-001",
                content="print(1)",
                language=ProgrammingLanguage.PYTHON,
                start_line=1,
                end_line=1,
            )

    def test_empty_content_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            CodeChunk(
                chunk_id="chunk-001",
                source_file_id="file-001",
                repository_id="repo-001",
                content="",
                language=ProgrammingLanguage.PYTHON,
                start_line=1,
                end_line=1,
            )

    def test_invalid_line_range_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            CodeChunk(
                chunk_id="chunk-001",
                source_file_id="file-001",
                repository_id="repo-001",
                content="print(1)",
                language=ProgrammingLanguage.PYTHON,
                start_line=10,
                end_line=5,
            )

    def test_invalid_start_line_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            CodeChunk(
                chunk_id="chunk-001",
                source_file_id="file-001",
                repository_id="repo-001",
                content="print(1)",
                language=ProgrammingLanguage.PYTHON,
                start_line=0,
                end_line=1,
            )

    def test_chunk_serialization(self) -> None:
        chunk = self.create_chunk()
        data = chunk.to_dict()

        self.assertEqual(data["chunk_id"], "chunk-001")
        self.assertEqual(data["symbol_name"], "add")
        self.assertEqual(data["start_line"], 10)


class TestRetrievedChunk(unittest.TestCase):
    """Tests for RetrievedChunk."""

    def test_valid_retrieved_chunk(self) -> None:
        chunk = CodeChunk(
            chunk_id="chunk-001",
            source_file_id="file-001",
            repository_id="repo-001",
            content="print(1)",
            language=ProgrammingLanguage.PYTHON,
            start_line=1,
            end_line=1,
        )

        retrieved = RetrievedChunk(
            chunk=chunk,
            retrieval_score=0.91,
            rank=1,
            retriever_name="faiss",
        )

        self.assertEqual(retrieved.chunk.chunk_id, "chunk-001")
        self.assertEqual(retrieved.rank, 1)
        self.assertEqual(retrieved.retrieval_score, 0.91)

    def test_invalid_rank_is_rejected(self) -> None:
        chunk = CodeChunk(
            chunk_id="chunk-001",
            source_file_id="file-001",
            repository_id="repo-001",
            content="print(1)",
            language=ProgrammingLanguage.PYTHON,
            start_line=1,
            end_line=1,
        )

        with self.assertRaises(ModelValidationError):
            RetrievedChunk(
                chunk=chunk,
                retrieval_score=0.91,
                rank=0,
            )


class TestGenerationRequest(unittest.TestCase):
    """Tests for GenerationRequest."""

    def test_valid_generation_request(self) -> None:
        request = GenerationRequest(
            request_id="request-001",
            task="code_generation",
            query="Create a secure password hashing function.",
            language=ProgrammingLanguage.PYTHON,
            model_name="test-model",
            provider_name="test-provider",
        )

        self.assertEqual(request.request_id, "request-001")
        self.assertTrue(request.defense_enabled)
        self.assertEqual(request.context, [])

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            GenerationRequest(
                request_id="request-001",
                task="code_generation",
                query="",
            )

    def test_nested_retrieved_context_is_supported(self) -> None:
        chunk = CodeChunk(
            chunk_id="chunk-001",
            source_file_id="file-001",
            repository_id="repo-001",
            content="def secure_hash(value):\n    return value",
            language=ProgrammingLanguage.PYTHON,
            start_line=1,
            end_line=2,
        )

        retrieved = RetrievedChunk(
            chunk=chunk,
            retrieval_score=0.95,
            rank=1,
        )

        request = GenerationRequest(
            request_id="request-001",
            task="code_generation",
            query="Create a secure hashing function.",
            context=[retrieved],
            language=ProgrammingLanguage.PYTHON,
        )

        self.assertEqual(len(request.context), 1)
        self.assertEqual(
            request.context[0].chunk.chunk_id,
            "chunk-001",
        )


class TestGenerationResult(unittest.TestCase):
    """Tests for GenerationResult."""

    def test_valid_generation_result(self) -> None:
        result = GenerationResult(
            request_id="request-001",
            generated_code="print('hello')",
            model_name="test-model",
            provider_name="test-provider",
            status=GenerationStatus.SUCCESS,
            prompt_tokens=100,
            completion_tokens=25,
            latency_ms=120.5,
            retrieved_chunk_ids=["chunk-001"],
        )

        self.assertEqual(result.status, GenerationStatus.SUCCESS)
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 25)
        self.assertEqual(result.latency_ms, 120.5)

    def test_negative_token_count_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            GenerationResult(
                request_id="request-001",
                generated_code="print(1)",
                model_name="test-model",
                provider_name="test-provider",
                prompt_tokens=-1,
            )

    def test_negative_latency_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            GenerationResult(
                request_id="request-001",
                generated_code="print(1)",
                model_name="test-model",
                provider_name="test-provider",
                latency_ms=-1,
            )


class TestSecurityFinding(unittest.TestCase):
    """Tests for SecurityFinding."""

    def test_valid_security_finding(self) -> None:
        finding = SecurityFinding(
            finding_id="finding-001",
            rule_id="SEC001",
            title="Potential command injection",
            description="User-controlled input reaches a shell command.",
            severity=SecuritySeverity.HIGH,
            decision=SecurityDecision.REJECT,
            file_path="src/example.py",
            start_line=10,
            end_line=10,
            evidence="subprocess.call(user_input, shell=True)",
            analyzer="static-analyzer",
            confidence=0.94,
        )

        self.assertEqual(finding.severity, SecuritySeverity.HIGH)
        self.assertEqual(finding.decision, SecurityDecision.REJECT)
        self.assertEqual(finding.confidence, 0.94)

    def test_invalid_confidence_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            SecurityFinding(
                finding_id="finding-001",
                rule_id="SEC001",
                title="Test finding",
                description="Test description.",
                severity=SecuritySeverity.MEDIUM,
                decision=SecurityDecision.FLAG,
                confidence=1.5,
            )

    def test_invalid_line_range_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            SecurityFinding(
                finding_id="finding-001",
                rule_id="SEC001",
                title="Test finding",
                description="Test description.",
                severity=SecuritySeverity.MEDIUM,
                decision=SecurityDecision.FLAG,
                start_line=20,
                end_line=10,
            )


class TestExperimentConfig(unittest.TestCase):
    """Tests for ExperimentConfig."""

    def test_valid_experiment_config(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            condition=ExperimentCondition.CLEAN_NO_DEFENSE,
            repository_id="repo-clean-001",
            task_type="code_generation",
            language=ProgrammingLanguage.PYTHON,
            model_name="test-model",
            retriever_name="faiss",
            top_k=5,
            defense_layers=[],
            seed=42,
        )

        self.assertEqual(
            config.condition,
            ExperimentCondition.CLEAN_NO_DEFENSE,
        )
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.seed, 42)

    def test_invalid_top_k_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            ExperimentConfig(
                experiment_id="exp-001",
                condition=ExperimentCondition.CLEAN_NO_DEFENSE,
                repository_id="repo-clean-001",
                task_type="code_generation",
                language=ProgrammingLanguage.PYTHON,
                model_name="test-model",
                retriever_name="faiss",
                top_k=0,
            )


class TestExperimentResult(unittest.TestCase):
    """Tests for ExperimentResult."""

    def create_finding(self) -> SecurityFinding:
        return SecurityFinding(
            finding_id="finding-001",
            rule_id="SEC001",
            title="Potential vulnerability",
            description="Security issue detected.",
            severity=SecuritySeverity.HIGH,
            decision=SecurityDecision.FLAG,
            confidence=0.90,
        )

    def test_valid_experiment_result(self) -> None:
        result = ExperimentResult(
            experiment_id="exp-001",
            condition=ExperimentCondition.POISONED_DEFENSE,
            request_id="request-001",
            poison_present=True,
            poison_retrieved=True,
            poison_in_context=True,
            generation_changed=True,
            vulnerability_introduced=False,
            poison_retrieval_rate=0.75,
            attack_success_rate=0.20,
            vulnerability_introduction_rate=0.10,
            utility_score=0.88,
            latency_ms=250.0,
            findings=[self.create_finding()],
        )

        self.assertTrue(result.poison_present)
        self.assertTrue(result.poison_retrieved)
        self.assertTrue(result.poison_in_context)
        self.assertTrue(result.generation_changed)
        self.assertFalse(result.vulnerability_introduced)

        self.assertEqual(result.poison_retrieval_rate, 0.75)
        self.assertEqual(result.attack_success_rate, 0.20)
        self.assertEqual(
            result.vulnerability_introduction_rate,
            0.10,
        )
        self.assertEqual(result.utility_score, 0.88)

        self.assertEqual(len(result.findings), 1)

    def test_metrics_must_be_between_zero_and_one(self) -> None:
        with self.assertRaises(ModelValidationError):
            ExperimentResult(
                experiment_id="exp-001",
                condition=ExperimentCondition.CLEAN_DEFENSE,
                request_id="request-001",
                poison_present=False,
                poison_retrieved=False,
                poison_in_context=False,
                generation_changed=False,
                vulnerability_introduced=False,
                attack_success_rate=1.1,
            )

    def test_negative_latency_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            ExperimentResult(
                experiment_id="exp-001",
                condition=ExperimentCondition.CLEAN_DEFENSE,
                request_id="request-001",
                poison_present=False,
                poison_retrieved=False,
                poison_in_context=False,
                generation_changed=False,
                vulnerability_introduced=False,
                latency_ms=-10,
            )

    def test_timestamp_is_timezone_aware(self) -> None:
        result = ExperimentResult(
            experiment_id="exp-001",
            condition=ExperimentCondition.CLEAN_DEFENSE,
            request_id="request-001",
            poison_present=False,
            poison_retrieved=False,
            poison_in_context=False,
            generation_changed=False,
            vulnerability_introduced=False,
        )

        self.assertIsInstance(result.created_at, datetime)
        self.assertIsNotNone(result.created_at.tzinfo)
        self.assertEqual(result.created_at.tzinfo, timezone.utc)

    def test_experiment_result_serialization(self) -> None:
        result = ExperimentResult(
            experiment_id="exp-001",
            condition=ExperimentCondition.CLEAN_NO_DEFENSE,
            request_id="request-001",
            poison_present=False,
            poison_retrieved=False,
            poison_in_context=False,
            generation_changed=False,
            vulnerability_introduced=False,
        )

        data = result.to_dict()

        self.assertEqual(data["experiment_id"], "exp-001")
        self.assertEqual(
            data["condition"],
            ExperimentCondition.CLEAN_NO_DEFENSE,
        )
        self.assertIn("created_at", data)


if __name__ == "__main__":
    unittest.main()