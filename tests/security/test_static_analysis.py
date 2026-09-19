from __future__ import annotations

import pytest

from src.models import SecurityDecision, SecuritySeverity
from src.security.ast_analyzer import ASTAnalyzerError
from src.security.findings import (
    SecurityFinding,
    SecurityLocation,
    SecurityReport,
)
from src.security.semgrep_runner import SemgrepRunnerError
from src.security.static_analysis import (
    StaticAnalysisConfig,
    StaticAnalysisInputError,
    StaticSecurityAnalyzer,
)


def make_finding(
    *,
    rule_id: str = "TEST-001",
    title: str = "Test finding",
    decision: SecurityDecision = SecurityDecision.FLAG,
    severity: SecuritySeverity = SecuritySeverity.MEDIUM,
    confidence: float = 0.9,
    file_name: str = "example.py",
    line_start: int = 1,
) -> SecurityFinding:
    """Create a valid security finding for isolated orchestration tests."""

    return SecurityFinding(
        finding_id=f"finding-{rule_id}",
        category="test",
        rule_id=rule_id,
        title=title,
        description="Synthetic test finding.",
        severity=severity,
        confidence=confidence,
        decision=decision,
        location=SecurityLocation(
            file_name=file_name,
            line_start=line_start,
            line_end=line_start,
        ),
        evidence="synthetic evidence",
        remediation="Use safe implementation.",
        source="test",
    )


class FakeSemgrepRunner:
    """Minimal Semgrep runner used to test L5 orchestration."""

    def __init__(
        self,
        *,
        available: bool = True,
        report: SecurityReport | None = None,
        error: Exception | None = None,
    ) -> None:
        self.available = available
        self.report = report
        self.error = error

    def is_available(self) -> bool:
        return self.available

    def analyze_code(
        self,
        code: str,
        *,
        file_name: str,
        language: str,
    ) -> SecurityReport:
        if self.error is not None:
            raise self.error

        if self.report is None:
            return SecurityReport(
                findings=(),
                decision=SecurityDecision.PASS,
                analyzer_name="fake-semgrep",
                metadata={
                    "language": language,
                    "file_name": file_name,
                },
            )

        return self.report


class FailingASTAnalyzer:
    """Fake AST analyzer used for fail-closed testing."""

    def analyze(
        self,
        code: str,
        *,
        file_name: str | None = None,
    ) -> SecurityReport:
        raise ASTAnalyzerError("synthetic AST analyzer failure")


class TestStaticAnalysisConfig:
    """Tests for StaticAnalysisConfig."""

    def test_default_configuration(self) -> None:
        config = StaticAnalysisConfig()

        assert config.enable_ast is True
        assert config.enable_semgrep is True
        assert config.deduplicate_findings is True
        assert config.fail_closed_on_ast_error is True
        assert config.fail_closed_on_semgrep_error is False
        assert config.analyzer_name == "static-analysis"

    def test_empty_analyzer_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="analyzer_name"):
            StaticAnalysisConfig(analyzer_name="")

    def test_all_backends_disabled_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="At least one static analysis backend",
        ):
            StaticAnalysisConfig(
                enable_ast=False,
                enable_semgrep=False,
            )


class TestStaticSecurityAnalyzer:
    """Tests for the L5 static security orchestrator."""

    def test_clean_python_code_passes(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        result = analyzer.analyze_code(
            "def add(a, b):\n    return a + b\n",
            file_name="clean.py",
        )

        assert result.decision == SecurityDecision.PASS
        assert result.finding_count == 0
        assert result.analyzers_run == ("ast",)
        assert result.semgrep_available is False

    def test_eval_is_rejected(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        result = analyzer.analyze_code(
            "value = eval(user_input)\n",
            file_name="unsafe.py",
        )

        assert result.decision == SecurityDecision.REJECT
        assert result.finding_count >= 1
        assert any(
            finding.rule_id == "PY-EVAL-001"
            for finding in result.findings
        )

    def test_exec_is_rejected(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        result = analyzer.analyze_code(
            "exec(user_input)\n",
            file_name="unsafe.py",
        )

        assert result.decision == SecurityDecision.REJECT
        assert any(
            finding.rule_id == "PY-EXEC-001"
            for finding in result.findings
        )

    def test_os_system_is_rejected(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        result = analyzer.analyze_code(
            "import os\nos.system(command)\n",
            file_name="unsafe.py",
        )

        assert result.decision == SecurityDecision.REJECT
        assert any(
            finding.rule_id == "PY-OS-001"
            for finding in result.findings
        )

    def test_hardcoded_secret_is_rejected(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        result = analyzer.analyze_code(
            'api_key = "super-secret-value"\n',
            file_name="secret.py",
        )

        assert result.decision == SecurityDecision.REJECT
        assert any(
            finding.rule_id == "PY-SECRET-001"
            for finding in result.findings
        )

    def test_semgrep_unavailable_does_not_break_ast_analysis(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            semgrep_runner=FakeSemgrepRunner(available=False),
        )

        result = analyzer.analyze_code(
            "eval(user_input)\n",
            file_name="unsafe.py",
        )

        assert result.decision == SecurityDecision.REJECT
        assert result.semgrep_available is False
        assert result.analyzers_run == ("ast",)
        assert result.analyzer_errors == ()

    def test_semgrep_report_is_combined_with_ast_report(self) -> None:
        semgrep_finding = make_finding(
            rule_id="SEMGREP-001",
            title="Semgrep test finding",
            decision=SecurityDecision.FLAG,
        )

        semgrep_report = SecurityReport(
            findings=(semgrep_finding,),
            decision=SecurityDecision.FLAG,
            analyzer_name="fake-semgrep",
        )

        analyzer = StaticSecurityAnalyzer(
            semgrep_runner=FakeSemgrepRunner(
                available=True,
                report=semgrep_report,
            )
        )

        result = analyzer.analyze_code(
            "def safe_function():\n    return 1\n",
            file_name="example.py",
        )

        assert result.decision == SecurityDecision.FLAG
        assert "ast" in result.analyzers_run
        assert "semgrep" in result.analyzers_run
        assert any(
            finding.rule_id == "SEMGREP-001"
            for finding in result.findings
        )

    def test_reject_takes_priority_over_flag(self) -> None:
        semgrep_finding = make_finding(
            rule_id="SEMGREP-FLAG",
            decision=SecurityDecision.FLAG,
        )

        semgrep_report = SecurityReport(
            findings=(semgrep_finding,),
            decision=SecurityDecision.FLAG,
            analyzer_name="fake-semgrep",
        )

        analyzer = StaticSecurityAnalyzer(
            semgrep_runner=FakeSemgrepRunner(
                available=True,
                report=semgrep_report,
            )
        )

        result = analyzer.analyze_code(
            "eval(user_input)\n",
            file_name="unsafe.py",
        )

        assert result.decision == SecurityDecision.REJECT

    def test_duplicate_findings_are_removed(self) -> None:
        finding = make_finding(
            rule_id="DUPLICATE-001",
            title="Duplicate finding",
            decision=SecurityDecision.FLAG,
        )

        first_report = SecurityReport(
            findings=(finding,),
            decision=SecurityDecision.FLAG,
            analyzer_name="first",
        )

        second_report = SecurityReport(
            findings=(finding,),
            decision=SecurityDecision.FLAG,
            analyzer_name="second",
        )

        class MultiReportRunner:
            def is_available(self) -> bool:
                return True

            def analyze_code(
                self,
                code: str,
                *,
                file_name: str,
                language: str,
            ) -> SecurityReport:
                return SecurityReport(
                    findings=(
                        *first_report.findings,
                        *second_report.findings,
                    ),
                    decision=SecurityDecision.FLAG,
                    analyzer_name="multi-semgrep",
                )

        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_ast=False,
                enable_semgrep=True,
            ),
            semgrep_runner=MultiReportRunner(),
        )

        result = analyzer.analyze_code(
            "print('hello')\n",
            file_name="example.py",
        )

        matching = [
            item
            for item in result.findings
            if item.rule_id == "DUPLICATE-001"
        ]

        assert len(matching) == 1
        assert result.decision == SecurityDecision.FLAG

    def test_ast_failure_fails_closed(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            ast_analyzer=FailingASTAnalyzer(),
            config=StaticAnalysisConfig(
                enable_semgrep=False,
                fail_closed_on_ast_error=True,
            ),
        )

        result = analyzer.analyze_code(
            "print('hello')\n",
            file_name="example.py",
        )

        assert result.decision == SecurityDecision.REJECT
        assert result.has_analyzer_errors is True
        assert result.metadata["analysis_failed"] is True
        assert any(
            "AST analyzer failed" in error
            for error in result.analyzer_errors
        )

    def test_semgrep_failure_can_be_non_fatal(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=True,
                fail_closed_on_semgrep_error=False,
            ),
            semgrep_runner=FakeSemgrepRunner(
                available=True,
                error=SemgrepRunnerError("synthetic failure"),
            ),
        )

        result = analyzer.analyze_code(
            "print('hello')\n",
            file_name="example.py",
        )

        assert result.decision == SecurityDecision.PASS
        assert result.analyzers_run == ("ast",)
        assert result.has_analyzer_errors is True

    def test_invalid_code_type_is_rejected(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        with pytest.raises(
            StaticAnalysisInputError,
            match="code must be a string",
        ):
            analyzer.analyze_code(123)  # type: ignore[arg-type]

    def test_empty_code_is_rejected(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        with pytest.raises(
            StaticAnalysisInputError,
            match="code must not be empty",
        ):
            analyzer.analyze_code("   ")

    def test_empty_file_name_is_rejected(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        with pytest.raises(
            StaticAnalysisInputError,
            match="file_name must be a non-empty string",
        ):
            analyzer.analyze_code(
                "print('hello')\n",
                file_name="",
            )

    def test_empty_language_is_rejected(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        with pytest.raises(
            StaticAnalysisInputError,
            match="language must be a non-empty string",
        ):
            analyzer.analyze_code(
                "print('hello')\n",
                language="",
            )

    def test_unsupported_language_skips_ast(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_ast=True,
                enable_semgrep=False,
            )
        )

        result = analyzer.analyze_code(
            "console.log('hello');\n",
            file_name="example.js",
            language="javascript",
        )

        assert result.decision == SecurityDecision.REJECT
        assert result.analyzers_run == ()
        assert result.has_analyzer_errors is True
        assert "unsupported language" in result.analyzer_errors[0]

    def test_batch_analysis_returns_one_result_per_item(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        results = analyzer.analyze_many(
            (
                ("one.py", "def one():\n    return 1\n"),
                ("two.py", "def two():\n    return 2\n"),
                ("three.py", "eval(value)\n"),
            )
        )

        assert len(results) == 3
        assert results[0].decision == SecurityDecision.PASS
        assert results[1].decision == SecurityDecision.PASS
        assert results[2].decision == SecurityDecision.REJECT

    def test_invalid_batch_item_is_rejected(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        with pytest.raises(
            StaticAnalysisInputError,
            match=r"\(file_name, code\) tuple",
        ):
            analyzer.analyze_many(
                (
                    ("valid.py", "print('hello')\n"),
                    ("invalid.py",),
                )  # type: ignore[list-item]
            )

    def test_metadata_is_preserved(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        result = analyzer.analyze_code(
            "print('hello')\n",
            file_name="example.py",
            metadata={
                "request_id": "req-001",
                "source": "generated",
            },
        )

        assert result.metadata["request_id"] == "req-001"
        assert result.metadata["source"] == "generated"
        assert result.metadata["language"] == "python"
        assert result.metadata["semgrep_available"] is False

    def test_report_analyzer_name_is_static_analysis(self) -> None:
        analyzer = StaticSecurityAnalyzer(
            config=StaticAnalysisConfig(
                enable_semgrep=False,
            )
        )

        result = analyzer.analyze_code(
            "print('hello')\n",
            file_name="example.py",
        )

        assert result.report.analyzer_name == "static-analysis"