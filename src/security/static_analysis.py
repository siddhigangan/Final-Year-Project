from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from src.security.ast_analyzer import ASTAnalyzerError, ASTSecurityAnalyzer
from src.security.findings import (
    SecurityDecision,
    SecurityFinding,
    SecurityReport,
    combine_security_reports,
)
from src.security.semgrep_runner import SemgrepRunner, SemgrepRunnerError


class StaticAnalysisError(Exception):
    """Base exception for static security analysis failures."""


class StaticAnalysisInputError(StaticAnalysisError):
    """Raised when static analysis receives invalid input."""


@dataclass(frozen=True)
class StaticAnalysisConfig:
    """Configuration for the L5 static analysis orchestrator."""

    enable_ast: bool = True
    enable_semgrep: bool = True
    deduplicate_findings: bool = True
    fail_closed_on_ast_error: bool = True
    fail_closed_on_semgrep_error: bool = False
    analyzer_name: str = "static-analysis"

    def __post_init__(self) -> None:
        if not self.analyzer_name.strip():
            raise ValueError("analyzer_name must not be empty.")

        if not self.enable_ast and not self.enable_semgrep:
            raise ValueError(
                "At least one static analysis backend must be enabled."
            )


@dataclass(frozen=True)
class StaticAnalysisResult:
    """Result returned by the L5 orchestration layer."""

    report: SecurityReport
    ast_enabled: bool
    semgrep_enabled: bool
    semgrep_available: bool
    analyzers_run: tuple[str, ...] = ()
    analyzer_errors: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def decision(self) -> SecurityDecision:
        """Return the final security decision."""

        return self.report.decision

    @property
    def findings(self) -> tuple[SecurityFinding, ...]:
        """Return all findings."""

        return self.report.findings

    @property
    def finding_count(self) -> int:
        """Return the number of findings."""

        return len(self.report.findings)

    @property
    def has_findings(self) -> bool:
        """Return whether any findings were produced."""

        return bool(self.report.findings)

    @property
    def has_analyzer_errors(self) -> bool:
        """Return whether any analyzer failed."""

        return bool(self.analyzer_errors)


class StaticSecurityAnalyzer:
    """
    L5 orchestration layer for static security analysis.

    AST analysis is the deterministic built-in backend.
    Semgrep is optional and is used when its CLI is available.
    """

    def __init__(
        self,
        config: StaticAnalysisConfig | None = None,
        *,
        ast_analyzer: ASTSecurityAnalyzer | None = None,
        semgrep_runner: SemgrepRunner | None = None,
    ) -> None:
        self._config = config or StaticAnalysisConfig()

        self._ast_analyzer = (
            ast_analyzer
            if ast_analyzer is not None
            else ASTSecurityAnalyzer()
        )

        self._semgrep_runner = (
            semgrep_runner
            if semgrep_runner is not None
            else SemgrepRunner()
        )

    @property
    def config(self) -> StaticAnalysisConfig:
        """Return the current L5 configuration."""

        return self._config

    @property
    def ast_analyzer(self) -> ASTSecurityAnalyzer:
        """Return the AST security analyzer."""

        return self._ast_analyzer

    @property
    def semgrep_runner(self) -> SemgrepRunner:
        """Return the Semgrep runner."""

        return self._semgrep_runner

    def analyze_code(
        self,
        code: str,
        *,
        file_name: str = "generated_code.py",
        language: str = "python",
        metadata: dict[str, Any] | None = None,
    ) -> StaticAnalysisResult:
        """
        Analyze source code using all enabled L5 backends.

        The AST analyzer currently supports Python analysis.
        Semgrep is optional and is executed only when available.
        """

        if not isinstance(code, str):
            raise StaticAnalysisInputError("code must be a string.")

        if not code.strip():
            raise StaticAnalysisInputError("code must not be empty.")

        if not isinstance(file_name, str) or not file_name.strip():
            raise StaticAnalysisInputError(
                "file_name must be a non-empty string."
            )

        if not isinstance(language, str) or not language.strip():
            raise StaticAnalysisInputError(
                "language must be a non-empty string."
            )

        normalized_language = language.strip().lower()

        reports: list[SecurityReport] = []
        analyzers_run: list[str] = []
        analyzer_errors: list[str] = []

        if self._config.enable_ast:
            if normalized_language in {"python", "py"}:
                try:
                    ast_report = self._run_ast(
                        code,
                        file_name=file_name,
                    )
                    reports.append(ast_report)
                    analyzers_run.append("ast")
                except ASTAnalyzerError as exc:
                    message = f"AST analyzer failed: {exc}"
                    analyzer_errors.append(message)

                    if self._config.fail_closed_on_ast_error:
                        return self._build_failure_result(
                            analyzers_run=analyzers_run,
                            analyzer_errors=analyzer_errors,
                            metadata=metadata,
                        )
            else:
                analyzer_errors.append(
                    f"AST analyzer skipped: unsupported language "
                    f"'{language}'."
                )

        semgrep_available = False

        if self._config.enable_semgrep:
            try:
                semgrep_available = self._semgrep_runner.is_available()

                if semgrep_available:
                    semgrep_report = self._run_semgrep(
                        code,
                        file_name=file_name,
                        language=normalized_language,
                    )
                    reports.append(semgrep_report)
                    analyzers_run.append("semgrep")
            except SemgrepRunnerError as exc:
                message = f"Semgrep analyzer failed: {exc}"
                analyzer_errors.append(message)

                if self._config.fail_closed_on_semgrep_error:
                    return self._build_failure_result(
                        analyzers_run=analyzers_run,
                        analyzer_errors=analyzer_errors,
                        metadata=metadata,
                    )

        if not reports:
            if analyzer_errors:
                return self._build_failure_result(
                    analyzers_run=analyzers_run,
                    analyzer_errors=analyzer_errors,
                    metadata=metadata,
                )

            raise StaticAnalysisError(
                "No static analysis backend produced a report."
            )

        combined = combine_security_reports(reports)

        if self._config.deduplicate_findings:
            combined = self._deduplicate_report(combined)

        combined = self._add_metadata(
            combined,
            metadata=metadata,
            analyzers_run=analyzers_run,
            analyzer_errors=analyzer_errors,
            semgrep_available=semgrep_available,
            language=normalized_language,
        )

        return StaticAnalysisResult(
            report=combined,
            ast_enabled=self._config.enable_ast,
            semgrep_enabled=self._config.enable_semgrep,
            semgrep_available=semgrep_available,
            analyzers_run=tuple(analyzers_run),
            analyzer_errors=tuple(analyzer_errors),
            metadata={
                **(metadata or {}),
                "analyzers_run": tuple(analyzers_run),
                "analyzer_errors": tuple(analyzer_errors),
                "semgrep_available": semgrep_available,
                "language": normalized_language,
            },
        )

    def analyze_many(
        self,
        code_items: Iterable[tuple[str, str]],
        *,
        language: str = "python",
    ) -> tuple[StaticAnalysisResult, ...]:
        """
        Analyze multiple code items.

        Each item must be a `(file_name, code)` tuple.
        """

        if isinstance(code_items, (str, bytes)):
            raise StaticAnalysisInputError(
                "code_items must be an iterable of "
                "(file_name, code) tuples."
            )

        try:
            items = tuple(code_items)
        except TypeError as exc:
            raise StaticAnalysisInputError(
                "code_items must be iterable."
            ) from exc

        results: list[StaticAnalysisResult] = []

        for item in items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise StaticAnalysisInputError(
                    "Each code item must be a (file_name, code) tuple."
                )

            file_name, code = item

            results.append(
                self.analyze_code(
                    code,
                    file_name=file_name,
                    language=language,
                )
            )

        return tuple(results)

    def _run_ast(
        self,
        code: str,
        *,
        file_name: str,
    ) -> SecurityReport:
        """Run the repository's Python AST security analyzer."""

        return self._ast_analyzer.analyze(
            code,
            file_name=file_name,
        )

    def _run_semgrep(
        self,
        code: str,
        *,
        file_name: str,
        language: str,
    ) -> SecurityReport:
        """Run Semgrep through the configured repository runner."""

        return self._semgrep_runner.analyze_code(
            code,
            file_name=file_name,
            language=language,
        )

    def _deduplicate_report(
        self,
        report: SecurityReport,
    ) -> SecurityReport:
        """Remove duplicate findings produced by multiple analyzers."""

        unique: dict[tuple[Any, ...], SecurityFinding] = {}

        for finding in report.findings:
            key = (
                finding.rule_id,
                finding.title,
                finding.location.file_name,
                finding.location.line_start,
                finding.location.line_end,
                finding.category,
            )

            existing = unique.get(key)

            if existing is None:
                unique[key] = finding
                continue

            if finding.confidence > existing.confidence:
                unique[key] = finding

        findings = tuple(unique.values())
        decision = self._aggregate_decision(findings)

        return SecurityReport(
            findings=findings,
            decision=decision,
            analyzer_name=report.analyzer_name,
            metadata={
                **report.metadata,
                "deduplicated": True,
                "original_finding_count": len(report.findings),
                "deduplicated_finding_count": len(findings),
            },
        )

    @staticmethod
    def _aggregate_decision(
        findings: Iterable[SecurityFinding],
    ) -> SecurityDecision:
        """Aggregate findings into one final security decision."""

        findings = tuple(findings)

        if any(
            finding.decision == SecurityDecision.REJECT
            for finding in findings
        ):
            return SecurityDecision.REJECT

        if any(
            finding.decision == SecurityDecision.FLAG
            for finding in findings
        ):
            return SecurityDecision.FLAG

        return SecurityDecision.PASS

    @staticmethod
    def _add_metadata(
        report: SecurityReport,
        *,
        metadata: dict[str, Any] | None,
        analyzers_run: list[str],
        analyzer_errors: list[str],
        semgrep_available: bool,
        language: str,
    ) -> SecurityReport:
        """Add orchestration metadata to the combined report."""

        return SecurityReport(
            findings=report.findings,
            decision=report.decision,
            analyzer_name="static-analysis",
            metadata={
                **report.metadata,
                **(metadata or {}),
                "analyzers_run": tuple(analyzers_run),
                "analyzer_errors": tuple(analyzer_errors),
                "semgrep_available": semgrep_available,
                "language": language,
            },
        )

    @staticmethod
    def _build_failure_result(
        *,
        analyzers_run: list[str],
        analyzer_errors: list[str],
        metadata: dict[str, Any] | None,
    ) -> StaticAnalysisResult:
        """Build a fail-closed result when analysis cannot complete."""

        report = SecurityReport(
            findings=(),
            decision=SecurityDecision.REJECT,
            analyzer_name="static-analysis",
            metadata={
                **(metadata or {}),
                "analysis_failed": True,
                "analyzers_run": tuple(analyzers_run),
                "analyzer_errors": tuple(analyzer_errors),
            },
        )

        return StaticAnalysisResult(
            report=report,
            ast_enabled=True,
            semgrep_enabled=True,
            semgrep_available=False,
            analyzers_run=tuple(analyzers_run),
            analyzer_errors=tuple(analyzer_errors),
            metadata={
                **(metadata or {}),
                "analysis_failed": True,
            },
        )