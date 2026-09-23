from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.models import (
    CodeChunk,
    ProgrammingLanguage,
    RetrievedChunk,
    SecurityDecision,
    SecuritySeverity,
)
from src.retrieval.context_builder import BuiltContext, ContextBuilder
from src.security.anomaly_detection import (
    AnomalyBatchAssessment,
    AnomalyDetector,
)
from src.security.context_validation import (
    ContextValidationResult,
    ContextValidator,
)
from src.security.decision import SecurityDecisionEngine
from src.security.findings import (
    SecurityFinding,
    SecurityLocation,
    SecurityReport,
)
from src.security.instruction_separator import (
    InstructionDataSeparator,
    SeparatedContent,
)
from src.security.risk_engine import (
    RiskAssessment,
    SecurityRiskEngine,
)
from src.security.static_analysis import (
    StaticAnalysisResult,
    StaticSecurityAnalyzer,
)
from src.security.trust_scoring import (
    SourceTrustScorer,
    TrustBatchAssessment,
)


class DefensePipelineError(RuntimeError):
    """Base exception for defense-pipeline failures."""


class DefensePipelineConfigurationError(ValueError):
    """Raised when defense-pipeline configuration is invalid."""


class DefensePipelineInputError(DefensePipelineError):
    """Raised when pipeline input is invalid."""


@dataclass(frozen=True)
class DefensePipelineConfig:
    """Configuration for the unified defense pipeline."""

    enabled: bool = True
    fail_closed: bool = True
    build_safe_context: bool = True
    analyze_generated_code: bool = False

    analyzer_name: str = "unified-defense"

    trust_threshold: float = 0.50
    anomaly_threshold: float = 0.50

    include_static_findings: bool = True

    generated_code_language: str = "python"
    generated_code_file_name: str = "generated_code.py"

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.analyzer_name.strip():
            raise DefensePipelineConfigurationError(
                "analyzer_name must not be empty."
            )

        if not 0.0 <= self.trust_threshold <= 1.0:
            raise DefensePipelineConfigurationError(
                "trust_threshold must be between 0 and 1."
            )

        if not 0.0 <= self.anomaly_threshold <= 1.0:
            raise DefensePipelineConfigurationError(
                "anomaly_threshold must be between 0 and 1."
            )

        if not self.generated_code_language.strip():
            raise DefensePipelineConfigurationError(
                "generated_code_language must not be empty."
            )

        if not self.generated_code_file_name.strip():
            raise DefensePipelineConfigurationError(
                "generated_code_file_name must not be empty."
            )


@dataclass
class DefensePipelineResult:
    """Complete result produced by the defense pipeline."""

    query: str | None = None
    input_chunk_count: int = 0

    trust: TrustBatchAssessment | None = None
    anomaly: AnomalyBatchAssessment | None = None
    context_validation: ContextValidationResult | None = None

    instruction_results: tuple[SeparatedContent, ...] = ()

    safe_context: BuiltContext | None = None

    static_analysis: StaticAnalysisResult | None = None

    decision_report: SecurityReport | None = None
    risk_assessment: RiskAssessment | None = None

    final_decision: SecurityDecision = SecurityDecision.PASS

    blocked: bool = False
    requires_review: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        """Return whether the pipeline considers the result safe."""
        return (
            self.final_decision == SecurityDecision.PASS
            and not self.blocked
        )

    @property
    def finding_count(self) -> int:
        """Return the number of security findings."""
        if self.decision_report is None:
            return 0

        return self.decision_report.finding_count

    @property
    def blocking_findings(self) -> tuple[SecurityFinding, ...]:
        """Return findings that block execution or usage."""
        if self.decision_report is None:
            return ()

        return tuple(self.decision_report.blocking_findings)


class DefensePipeline:
    """
    Unified L1-L5 security pipeline for SecureCodeRAG.

    Layers:

    L1 - Source trust scoring
    L2 - Suspicious/anomaly detection
    L3 - Context validation
    L4 - Instruction/data separation
    L5 - Static security analysis
    """

    def __init__(
        self,
        config: DefensePipelineConfig | None = None,
        *,
        context_builder: ContextBuilder | None = None,
        trust_scorer: SourceTrustScorer | None = None,
        anomaly_detector: AnomalyDetector | None = None,
        context_validator: ContextValidator | None = None,
        instruction_separator: InstructionDataSeparator | None = None,
        static_analyzer: StaticSecurityAnalyzer | None = None,
        decision_engine: SecurityDecisionEngine | None = None,
        risk_engine: SecurityRiskEngine | None = None,
    ) -> None:
        self._config = config or DefensePipelineConfig()

        self._context_builder = (
            context_builder or ContextBuilder()
        )

        self._trust_scorer = (
            trust_scorer or SourceTrustScorer()
        )

        self._anomaly_detector = (
            anomaly_detector or AnomalyDetector()
        )

        self._context_validator = (
            context_validator or ContextValidator()
        )

        self._instruction_separator = (
            instruction_separator or InstructionDataSeparator()
        )

        self._static_analyzer = (
            static_analyzer or StaticSecurityAnalyzer()
        )

        self._decision_engine = (
            decision_engine or SecurityDecisionEngine()
        )

        self._risk_engine = (
            risk_engine or SecurityRiskEngine()
        )

    @property
    def config(self) -> DefensePipelineConfig:
        """Return the active pipeline configuration."""
        return self._config

    @property
    def analyzer_name(self) -> str:
        """Return the configured analyzer name."""
        return self._config.analyzer_name

    def __repr__(self) -> str:
        """Return a concise pipeline representation."""
        return (
            "DefensePipeline("
            f"analyzer_name={self._config.analyzer_name!r}, "
            f"enabled={self._config.enabled!r}, "
            f"fail_closed={self._config.fail_closed!r}"
            ")"
        )

    def analyze_context(
        self,
        *,
        query: str,
        context: BuiltContext,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> DefensePipelineResult:
        """Run L1-L5 over retrieved context."""
        self._validate_context_inputs(
            query=query,
            context=context,
            retrieved_chunks=retrieved_chunks,
        )

        chunk_list = list(retrieved_chunks)

        if not self._config.enabled:
            return DefensePipelineResult(
                query=query,
                input_chunk_count=len(chunk_list),
                safe_context=context,
                final_decision=SecurityDecision.PASS,
                metadata={
                    "pipeline_enabled": False,
                    "analyzer_name": self._config.analyzer_name,
                    "context_safe_to_use": True,
                },
            )

        try:
            trust_result = (
                self._trust_scorer.score_retrieved_chunks(
                    chunk_list
                )
            )

            anomaly_result = (
                self._anomaly_detector.analyze_retrieved_chunks(
                    chunk_list
                )
            )

            validation_result = self._context_validator.validate(
                context,
                chunk_list,
            )

            instruction_results = tuple(
                self._instruction_separator.separate_chunk(
                    retrieved_chunk
                )
                for retrieved_chunk in chunk_list
            )

            safe_context = (
                self._build_safe_context(
                    query=query,
                    context=context,
                    retrieved_chunks=chunk_list,
                    instruction_results=instruction_results,
                    validation_result=validation_result,
                )
                if self._config.build_safe_context
                else context
            )

            static_result = self._analyze_context_code(
                context=safe_context,
                retrieved_chunks=chunk_list,
            )

            findings = self._collect_findings(
                trust_result=trust_result,
                anomaly_result=anomaly_result,
                instruction_results=instruction_results,
                static_result=static_result,
            )

            decision_report = self._decision_engine.evaluate(
                findings
            )

            risk_assessment = self._risk_engine.assess_report(
                decision_report
            )

            final_decision = risk_assessment.decision

            return DefensePipelineResult(
                query=query,
                input_chunk_count=len(chunk_list),
                trust=trust_result,
                anomaly=anomaly_result,
                context_validation=validation_result,
                instruction_results=instruction_results,
                static_analysis=static_result,
                decision_report=decision_report,
                risk_assessment=risk_assessment,
                safe_context=safe_context,
                final_decision=final_decision,
                blocked=(
                    final_decision
                    == SecurityDecision.REJECT
                ),
                requires_review=(
                    final_decision
                    == SecurityDecision.FLAG
                ),
                metadata={
                    "pipeline_enabled": True,
                    "analyzer_name": self._config.analyzer_name,
                    "context_safe_to_use": (
                        validation_result.safe_to_use
                    ),
                    "input_chunk_count": len(chunk_list),
                    "instruction_signal_count": sum(
                        result.signal_count
                        for result in instruction_results
                    ),
                    "anomaly_count": (
                        anomaly_result.anomalous_count
                    ),
                    "trust_minimum_score": (
                        trust_result.minimum_score
                    ),
                    "trust_maximum_score": (
                        trust_result.maximum_score
                    ),
                    "static_analysis_performed": (
                        static_result is not None
                    ),
                },
            )

        except DefensePipelineError:
            raise

        except Exception as exc:
            if self._config.fail_closed:
                raise DefensePipelineError(
                    f"Defense pipeline failed: {exc}"
                ) from exc

            return DefensePipelineResult(
                query=query,
                input_chunk_count=len(chunk_list),
                safe_context=None,
                final_decision=SecurityDecision.FLAG,
                requires_review=True,
                metadata={
                    "pipeline_enabled": True,
                    "analyzer_name": self._config.analyzer_name,
                    "pipeline_error": str(exc),
                    "fail_closed": False,
                },
            )

    def analyze_generated_code(
        self,
        *,
        code: str,
        file_name: str | None = None,
        language: str | None = None,
        query: str | None = None,
    ) -> DefensePipelineResult:
        """Analyze generated code using L5 static security analysis."""
        if not isinstance(code, str) or not code.strip():
            raise DefensePipelineInputError(
                "Generated code must be a non-empty string."
            )

        if not self._config.enabled:
            return DefensePipelineResult(
                query=query,
                input_chunk_count=0,
                final_decision=SecurityDecision.PASS,
                metadata={
                    "pipeline_enabled": False,
                    "analyzer_name": self._config.analyzer_name,
                    "generated_code_analysis": False,
                },
            )

        effective_file_name = (
            file_name or self._config.generated_code_file_name
        )

        effective_language = (
            language or self._config.generated_code_language
        )

        try:
            static_result = self._static_analyzer.analyze_code(
                code=code,
                file_name=effective_file_name,
                language=effective_language,
                metadata={
                    "generated_code": True,
                    "query": query,
                },
            )

            findings = (
                tuple(static_result.findings)
                if self._config.include_static_findings
                else ()
            )

            decision_report = self._decision_engine.evaluate(
                findings
            )

            risk_assessment = self._risk_engine.assess_report(
                decision_report
            )

            final_decision = risk_assessment.decision

            return DefensePipelineResult(
                query=query,
                input_chunk_count=0,
                static_analysis=static_result,
                decision_report=decision_report,
                risk_assessment=risk_assessment,
                final_decision=final_decision,
                blocked=(
                    final_decision
                    == SecurityDecision.REJECT
                ),
                requires_review=(
                    final_decision
                    == SecurityDecision.FLAG
                ),
                metadata={
                    "pipeline_enabled": True,
                    "analyzer_name": self._config.analyzer_name,
                    "generated_code_analysis": True,
                    "file_name": effective_file_name,
                    "language": effective_language,
                    "finding_count": len(findings),
                },
            )

        except DefensePipelineError:
            raise

        except Exception as exc:
            if self._config.fail_closed:
                raise DefensePipelineError(
                    f"Generated-code analysis failed: {exc}"
                ) from exc

            return DefensePipelineResult(
                query=query,
                input_chunk_count=0,
                final_decision=SecurityDecision.FLAG,
                requires_review=True,
                metadata={
                    "pipeline_enabled": True,
                    "analyzer_name": self._config.analyzer_name,
                    "generated_code_analysis": True,
                    "pipeline_error": str(exc),
                    "fail_closed": False,
                },
            )

    def _validate_context_inputs(
        self,
        *,
        query: str,
        context: BuiltContext,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> None:
        """Validate context-analysis inputs."""
        if not isinstance(query, str) or not query.strip():
            raise DefensePipelineInputError(
                "query must be a non-empty string."
            )

        if not isinstance(context, BuiltContext):
            raise DefensePipelineInputError(
                "context must be a BuiltContext instance."
            )

        if retrieved_chunks is None:
            raise DefensePipelineInputError(
                "retrieved_chunks must not be None."
            )

        for item in retrieved_chunks:
            if not isinstance(item, RetrievedChunk):
                raise DefensePipelineInputError(
                    "retrieved_chunks must contain only "
                    "RetrievedChunk instances."
                )

    def _build_safe_context(
        self,
        *,
        query: str,
        context: BuiltContext,
        retrieved_chunks: Sequence[RetrievedChunk],
        instruction_results: Sequence[SeparatedContent],
        validation_result: ContextValidationResult,
    ) -> BuiltContext:
        """Build sanitized context after L3/L4."""

        if not validation_result.safe_to_use:
            return BuiltContext(
                text="",
                sources=(),
                query=query,
                total_characters=0,
                total_lines=0,
                truncated=False,
                metadata={
                    "sanitized": True,
                    "reason": "context_validation_failed",
                },
            )

        safe_chunks: list[RetrievedChunk] = []

        for retrieved_chunk, separated in zip(
            retrieved_chunks,
            instruction_results,
        ):
            safe_content = separated.safe_content

            if not safe_content.strip():
                continue

            original = retrieved_chunk.chunk

            sanitized_chunk = CodeChunk(
                chunk_id=original.chunk_id,
                source_file_id=original.source_file_id,
                repository_id=original.repository_id,
                content=safe_content,
                language=original.language,
                start_line=original.start_line,
                end_line=original.end_line,
                symbol_name=original.symbol_name,
                symbol_type=original.symbol_type,
                ast_node_type=original.ast_node_type,
                parent_symbol=original.parent_symbol,
                is_documentation=original.is_documentation,
                trust=original.trust,
                metadata={
                    **original.metadata,
                    "sanitized_by_defense_pipeline": True,
                },
            )

            safe_chunks.append(
                RetrievedChunk(
                    chunk=sanitized_chunk,
                    retrieval_score=retrieved_chunk.retrieval_score,
                    rank=retrieved_chunk.rank,
                    retriever_name=retrieved_chunk.retriever_name,
                    rerank_score=retrieved_chunk.rerank_score,
                    retrieval_metadata={
                        **retrieved_chunk.retrieval_metadata,
                        "sanitized_by_defense_pipeline": True,
                    },
                )
            )

        if not safe_chunks:
            return BuiltContext(
                text="",
                sources=(),
                query=query,
                total_characters=0,
                total_lines=0,
                truncated=False,
                metadata={
                    "sanitized": True,
                    "reason": "no_safe_chunks",
                },
            )

        return self._context_builder.build(
            safe_chunks,
            query=query,
        )

    def _analyze_context_code(
        self,
        *,
        context: BuiltContext | None,
        retrieved_chunks: Sequence[RetrievedChunk] = (),
    ) -> StaticAnalysisResult | None:
        """Run L5 against safe retrieved Python code."""
        if context is None or not context.text.strip():
            return None

        python_parts: list[str] = []

        for index, retrieved_chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            language = self._language_value(
                retrieved_chunk.chunk.language
            )

            if language != ProgrammingLanguage.PYTHON.value:
                continue

            file_name = (
                retrieved_chunk.chunk.metadata.get(
                    "relative_path"
                )
                or retrieved_chunk.chunk.source_file_id
                or f"retrieved_source_{index}.py"
            )

            content = retrieved_chunk.chunk.content

            if not content.strip():
                continue

            python_parts.append(
                f"# Source: {file_name}\n{content}"
            )

        if not python_parts:
            return None

        combined_code = "\n\n".join(python_parts)

        return self._static_analyzer.analyze_code(
            code=combined_code,
            file_name="retrieved_context.py",
            language=ProgrammingLanguage.PYTHON.value,
            metadata={
                "source_count": len(python_parts),
                "pipeline_layer": "L5",
                "analyzer_name": self._config.analyzer_name,
            },
        )

    @staticmethod
    def _language_value(
        language: ProgrammingLanguage | str,
    ) -> str:
        """Normalize language enum/string values."""
        if isinstance(language, ProgrammingLanguage):
            return language.value

        return str(language).strip().lower()

    def _collect_findings(
        self,
        *,
        trust_result: TrustBatchAssessment,
        anomaly_result: AnomalyBatchAssessment,
        instruction_results: Sequence[SeparatedContent],
        static_result: StaticAnalysisResult | None,
    ) -> tuple[SecurityFinding, ...]:
        """Convert all defense-layer outputs into security findings."""
        findings: list[SecurityFinding] = []

        # L1: Source trust
        for assessment in trust_result.assessments:
            if assessment.score >= self._config.trust_threshold:
                continue

            severity = (
                SecuritySeverity.HIGH
                if assessment.score < 0.25
                else SecuritySeverity.MEDIUM
            )

            trust_value = getattr(
                assessment.trust,
                "value",
                str(assessment.trust),
            )

            findings.append(
                SecurityFinding(
                    finding_id=(
                        f"DEF-L1-TRUST-{assessment.chunk_id}"
                    ),
                    category="source_trust",
                    rule_id="DEF-L1-001",
                    title="Low-trust retrieved source",
                    description=(
                        "A retrieved chunk received a trust score "
                        "below the configured threshold."
                    ),
                    severity=severity,
                    confidence=1.0 - assessment.score,
                    decision=SecurityDecision.FLAG,
                    location=SecurityLocation(
                        file_name=(
                            getattr(
                                assessment,
                                "relative_path",
                                None,
                            )
                            or assessment.source_file_id
                        ),
                    ),
                    evidence=(
                        f"trust_score={assessment.score:.4f}"
                    ),
                    remediation=(
                        "Review source provenance before using "
                        "the retrieved content."
                    ),
                    source=self._config.analyzer_name,
                    metadata={
                        "trust_score": assessment.score,
                        "source_trust": trust_value,
                    },
                )
            )

        # L2: Anomaly detection
        for assessment in anomaly_result.assessments:
            if not assessment.anomalous:
                continue

            severity = (
                SecuritySeverity.HIGH
                if assessment.anomaly_score >= 0.80
                else SecuritySeverity.MEDIUM
            )

            findings.append(
                SecurityFinding(
                    finding_id=(
                        f"DEF-L2-ANOMALY-{assessment.chunk_id}"
                    ),
                    category="anomaly",
                    rule_id="DEF-L2-001",
                    title="Suspicious retrieved content",
                    description=(
                        "Retrieved content exceeded the configured "
                        "anomaly threshold."
                    ),
                    severity=severity,
                    confidence=assessment.anomaly_score,
                    decision=SecurityDecision.FLAG,
                    location=SecurityLocation(
                        file_name=assessment.source_file_id,
                    ),
                    evidence=(
                        f"anomaly_score="
                        f"{assessment.anomaly_score:.4f}"
                    ),
                    remediation=(
                        "Inspect the retrieved source and provenance "
                        "before passing it to the model."
                    ),
                    source=self._config.analyzer_name,
                    metadata={
                        "anomaly_score": assessment.anomaly_score,
                        "source_file_id": assessment.source_file_id,
                    },
                )
            )

        # L4: Instruction/data separation
        for index, separated in enumerate(
            instruction_results,
            start=1,
        ):
            if not separated.requires_review:
                continue

            relative_path = None

            if separated.metadata:
                relative_path = separated.metadata.get(
                    "relative_path"
                )

            findings.append(
                SecurityFinding(
                    finding_id=f"DEF-L4-INSTRUCTION-{index}",
                    category="instruction_separation",
                    rule_id="DEF-L4-001",
                    title="Instruction-like retrieved content",
                    description=(
                        "Retrieved content contains instruction-like "
                        "patterns that require review."
                    ),
                    severity=SecuritySeverity.HIGH,
                    confidence=separated.confidence,
                    decision=SecurityDecision.FLAG,
                    location=SecurityLocation(
                        file_name=relative_path,
                    ),
                    evidence=(
                        separated.original_content[:500]
                    ),
                    remediation=(
                        "Treat instruction-like retrieved text as "
                        "untrusted data rather than model instructions."
                    ),
                    source=self._config.analyzer_name,
                    metadata={
                        "classification": (
                            separated.classification.value
                        ),
                        "signal_count": separated.signal_count,
                    },
                )
            )

        # L5: Static security analysis
        if (
            self._config.include_static_findings
            and static_result is not None
        ):
            findings.extend(static_result.findings)

        return self._deduplicate_findings(findings)

    @staticmethod
    def _deduplicate_findings(
        findings: Sequence[SecurityFinding],
    ) -> tuple[SecurityFinding, ...]:
        """Remove duplicate findings by stable finding ID."""
        unique: dict[str, SecurityFinding] = {}

        for finding in findings:
            unique.setdefault(
                finding.finding_id,
                finding,
            )

        return tuple(unique.values())


__all__ = [
    "DefensePipeline",
    "DefensePipelineConfig",
    "DefensePipelineConfigurationError",
    "DefensePipelineError",
    "DefensePipelineInputError",
    "DefensePipelineResult",
]