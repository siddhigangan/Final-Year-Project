from __future__ import annotations

import json
import subprocess

import pytest

from src.models import SecurityDecision, SecuritySeverity
from src.security.findings import SecurityFinding, SecurityLocation
from src.security.semgrep_runner import (
    SemgrepConfig,
    SemgrepConfigurationError,
    SemgrepExecutionError,
    SemgrepRunner,
    SemgrepRunnerError,
)


def make_finding(decision: SecurityDecision) -> SecurityFinding:
    return SecurityFinding(
        finding_id="test-finding",
        category="test",
        rule_id="test-rule",
        title="Test finding",
        description="Test security finding",
        severity=(
            SecuritySeverity.HIGH
            if decision == SecurityDecision.REJECT
            else SecuritySeverity.MEDIUM
        ),
        confidence=0.9,
        decision=decision,
        location=SecurityLocation(
            file_name="example.py",
            line_start=1,
            line_end=1,
        ),
        evidence="test evidence",
        remediation="test remediation",
        source="test",
    )


class TestSemgrepConfig:
    def test_default_configuration(self):
        config = SemgrepConfig()

        assert config.executable == "semgrep"
        assert config.config == "auto"
        assert config.timeout_seconds == 60
        assert config.max_target_bytes == 2_000_000
        assert config.additional_args == ()

    def test_rejects_empty_executable(self):
        with pytest.raises(SemgrepConfigurationError):
            SemgrepConfig(executable="")

    def test_rejects_empty_config(self):
        with pytest.raises(SemgrepConfigurationError):
            SemgrepConfig(config="")

    def test_rejects_invalid_timeout(self):
        with pytest.raises(SemgrepConfigurationError):
            SemgrepConfig(timeout_seconds=0)

    def test_rejects_invalid_target_size(self):
        with pytest.raises(SemgrepConfigurationError):
            SemgrepConfig(max_target_bytes=0)

    def test_rejects_invalid_additional_argument(self):
        with pytest.raises(SemgrepConfigurationError):
            SemgrepConfig(additional_args=("valid", ""))


class TestSemgrepAvailability:
    def test_unavailable_executable(self, monkeypatch):
        runner = SemgrepRunner()

        monkeypatch.setattr(
            "src.security.semgrep_runner.shutil.which",
            lambda executable: None,
        )

        assert runner.is_available() is False

    def test_available_executable(self, monkeypatch):
        runner = SemgrepRunner()

        monkeypatch.setattr(
            "src.security.semgrep_runner.shutil.which",
            lambda executable: "C:\\Tools\\semgrep.exe",
        )

        assert runner.is_available() is True

    def test_version_returns_none_when_unavailable(self, monkeypatch):
        runner = SemgrepRunner()

        monkeypatch.setattr(
            "src.security.semgrep_runner.shutil.which",
            lambda executable: None,
        )

        assert runner.version() is None


class TestAnalyzeCode:
    def test_rejects_non_string_code(self):
        runner = SemgrepRunner()

        with pytest.raises(SemgrepRunnerError):
            runner.analyze_code(123)

    def test_rejects_empty_code(self):
        runner = SemgrepRunner()

        with pytest.raises(SemgrepRunnerError):
            runner.analyze_code("   ")

    def test_rejects_empty_file_name(self):
        runner = SemgrepRunner()

        with pytest.raises(SemgrepRunnerError):
            runner.analyze_code(
                "print('hello')",
                file_name="",
            )

    def test_rejects_oversized_code(self):
        runner = SemgrepRunner(
            SemgrepConfig(max_target_bytes=10)
        )

        with pytest.raises(SemgrepRunnerError):
            runner.analyze_code(
                "print('this is too large')",
            )

    def test_returns_unavailable_result(self, monkeypatch):
        runner = SemgrepRunner()

        monkeypatch.setattr(
            "src.security.semgrep_runner.shutil.which",
            lambda executable: None,
        )

        result = runner.analyze_code(
            "print('hello')",
            file_name="example.py",
            language="python",
        )

        assert result.available is False
        assert result.findings == ()
        assert result.error == "Semgrep executable was not found"


class TestAnalyzeFile:
    def test_rejects_missing_file(self, tmp_path):
        runner = SemgrepRunner()

        missing = tmp_path / "missing.py"

        with pytest.raises(SemgrepRunnerError):
            runner.analyze_file(missing)

    def test_rejects_directory(self, tmp_path):
        runner = SemgrepRunner()

        directory = tmp_path / "directory"
        directory.mkdir()

        with pytest.raises(SemgrepRunnerError):
            runner.analyze_file(directory)

    def test_rejects_oversized_file(self, tmp_path):
        runner = SemgrepRunner(
            SemgrepConfig(max_target_bytes=5)
        )

        target = tmp_path / "example.py"
        target.write_text(
            "print('too large')",
            encoding="utf-8",
        )

        with pytest.raises(SemgrepRunnerError):
            runner.analyze_file(target)

    def test_returns_unavailable_for_existing_file(
        self,
        tmp_path,
        monkeypatch,
    ):
        runner = SemgrepRunner()

        target = tmp_path / "example.py"
        target.write_text(
            "print('hello')",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "src.security.semgrep_runner.shutil.which",
            lambda executable: None,
        )

        result = runner.analyze_file(target)

        assert result.available is False
        assert result.findings == ()


class TestCommandConstruction:
    def test_builds_basic_command(self, tmp_path):
        runner = SemgrepRunner(
            SemgrepConfig(
                executable="semgrep",
                config="auto",
            )
        )

        target = tmp_path / "example.py"

        command = runner._build_command(target, language=None)

        assert command == [
            "semgrep",
            "--json",
            "--config",
            "auto",
            str(target),
        ]

    def test_builds_language_command(self, tmp_path):
        runner = SemgrepRunner()

        target = tmp_path / "example.py"

        command = runner._build_command(
            target,
            language="python",
        )

        assert command == [
            "semgrep",
            "--json",
            "--config",
            "auto",
            "--lang",
            "python",
            str(target),
        ]

    def test_includes_additional_arguments(self, tmp_path):
        runner = SemgrepRunner(
            SemgrepConfig(
                additional_args=(
                    "--no-git-ignore",
                    "--disable-version-check",
                )
            )
        )

        target = tmp_path / "example.py"

        command = runner._build_command(
            target,
            language="python",
        )

        assert "--no-git-ignore" in command
        assert "--disable-version-check" in command


class TestSemgrepParsing:
    def test_parses_high_severity_finding(self):
        runner = SemgrepRunner()

        payload = {
            "results": [
                {
                    "check_id": "python.lang.security.eval-detected",
                    "path": "example.py",
                    "start": {
                        "line": 4,
                        "col": 5,
                    },
                    "end": {
                        "line": 4,
                        "col": 20,
                    },
                    "extra": {
                        "message": "Dangerous eval usage",
                        "severity": "ERROR",
                        "lines": "eval(user_input)",
                        "metadata": {
                            "confidence": "HIGH",
                        },
                    },
                }
            ]
        }

        findings = runner._parse_findings(payload)

        assert len(findings) == 1

        finding = findings[0]

        assert (
            finding.rule_id
            == "python.lang.security.eval-detected"
        )
        assert finding.severity == SecuritySeverity.HIGH
        assert finding.confidence == 0.85
        assert finding.decision == SecurityDecision.REJECT
        assert finding.location.file_name == "example.py"
        assert finding.location.line_start == 4
        assert finding.location.line_end == 4
        assert finding.location.column_start == 5
        assert finding.location.column_end == 20
        assert finding.evidence == "eval(user_input)"
        assert finding.source == "semgrep"

    def test_parses_medium_severity_finding(self):
        runner = SemgrepRunner()

        payload = {
            "results": [
                {
                    "check_id": "test-rule",
                    "path": "example.py",
                    "start": {"line": 2},
                    "end": {"line": 2},
                    "extra": {
                        "message": "Potential issue",
                        "severity": "WARNING",
                        "metadata": {
                            "confidence": "MEDIUM",
                        },
                    },
                }
            ]
        }

        findings = runner._parse_findings(payload)

        assert len(findings) == 1
        assert findings[0].severity == SecuritySeverity.MEDIUM
        assert findings[0].confidence == 0.70
        assert findings[0].decision == SecurityDecision.FLAG

    def test_defaults_unknown_severity_to_medium(self):
        runner = SemgrepRunner()

        payload = {
            "results": [
                {
                    "check_id": "unknown-rule",
                    "path": "example.py",
                    "extra": {
                        "message": "Unknown severity",
                    },
                }
            ]
        }

        findings = runner._parse_findings(payload)

        assert findings[0].severity == SecuritySeverity.MEDIUM

    def test_defaults_unknown_confidence(self):
        runner = SemgrepRunner()

        payload = {
            "results": [
                {
                    "check_id": "unknown-rule",
                    "path": "example.py",
                    "extra": {
                        "message": "Unknown confidence",
                        "severity": "INFO",
                    },
                }
            ]
        }

        findings = runner._parse_findings(payload)

        assert findings[0].confidence == 0.70

    def test_confidence_numeric_value_is_clamped(self):
        runner = SemgrepRunner()

        payload = {
            "results": [
                {
                    "check_id": "test-rule",
                    "path": "example.py",
                    "extra": {
                        "message": "Test",
                        "severity": "LOW",
                        "metadata": {
                            "confidence": 2.0,
                        },
                    },
                }
            ]
        }

        findings = runner._parse_findings(payload)

        assert findings[0].confidence == 1.0

    def test_parses_multiple_findings(self):
        runner = SemgrepRunner()

        payload = {
            "results": [
                {
                    "check_id": "rule-one",
                    "path": "one.py",
                    "extra": {
                        "message": "Finding one",
                        "severity": "ERROR",
                    },
                },
                {
                    "check_id": "rule-two",
                    "path": "two.py",
                    "extra": {
                        "message": "Finding two",
                        "severity": "WARNING",
                    },
                },
            ]
        }

        findings = runner._parse_findings(payload)

        assert len(findings) == 2
        assert findings[0].rule_id == "rule-one"
        assert findings[1].rule_id == "rule-two"

    def test_rejects_invalid_results_field(self):
        runner = SemgrepRunner()

        with pytest.raises(SemgrepExecutionError):
            runner._parse_findings(
                {
                    "results": {},
                }
            )


class TestSemgrepExecution:
    def test_successful_execution_with_no_findings(
        self,
        tmp_path,
        monkeypatch,
    ):
        runner = SemgrepRunner()

        target = tmp_path / "example.py"
        target.write_text(
            "print('hello')",
            encoding="utf-8",
        )

        payload = {
            "results": [],
        }

        monkeypatch.setattr(
            "src.security.semgrep_runner.subprocess.run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=args[0],
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            ),
        )

        result = runner._run_target(target, language="python")

        assert result.available is True
        assert result.return_code == 0
        assert result.findings == ()
        assert result.error is None

    def test_successful_execution_with_findings(
        self,
        tmp_path,
        monkeypatch,
    ):
        runner = SemgrepRunner()

        target = tmp_path / "example.py"
        target.write_text(
            "eval(user_input)",
            encoding="utf-8",
        )

        payload = {
            "results": [
                {
                    "check_id": "eval-rule",
                    "path": str(target),
                    "start": {"line": 1},
                    "end": {"line": 1},
                    "extra": {
                        "message": "Avoid eval",
                        "severity": "ERROR",
                    },
                }
            ]
        }

        monkeypatch.setattr(
            "src.security.semgrep_runner.subprocess.run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=args[0],
                returncode=1,
                stdout=json.dumps(payload),
                stderr="",
            ),
        )

        result = runner._run_target(target, language="python")

        assert result.available is True
        assert result.return_code == 1
        assert len(result.findings) == 1
        assert result.findings[0].decision == SecurityDecision.REJECT
        assert result.error is None

    def test_handles_nonzero_execution_without_json(
        self,
        tmp_path,
        monkeypatch,
    ):
        runner = SemgrepRunner()

        target = tmp_path / "example.py"
        target.write_text(
            "print('hello')",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "src.security.semgrep_runner.subprocess.run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=args[0],
                returncode=2,
                stdout="",
                stderr="Semgrep failed",
            ),
        )

        result = runner._run_target(target, language="python")

        assert result.available is True
        assert result.return_code == 2
        assert result.findings == ()
        assert result.error == "Semgrep failed"

    def test_rejects_invalid_json(
        self,
        tmp_path,
        monkeypatch,
    ):
        runner = SemgrepRunner()

        target = tmp_path / "example.py"
        target.write_text(
            "print('hello')",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "src.security.semgrep_runner.subprocess.run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=args[0],
                returncode=0,
                stdout="{not valid json",
                stderr="",
            ),
        )

        with pytest.raises(SemgrepExecutionError):
            runner._run_target(target, language="python")

    def test_handles_timeout(
        self,
        tmp_path,
        monkeypatch,
    ):
        runner = SemgrepRunner()

        target = tmp_path / "example.py"
        target.write_text(
            "print('hello')",
            encoding="utf-8",
        )

        def raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(
                cmd=args[0],
                timeout=60,
            )

        monkeypatch.setattr(
            "src.security.semgrep_runner.subprocess.run",
            raise_timeout,
        )

        result = runner._run_target(target, language="python")

        assert result.available is True
        assert result.return_code is None
        assert result.error is not None
        assert "timed out" in result.error.lower()


class TestSecurityReport:
    def test_unavailable_semgrep_produces_pass_report(
        self,
        monkeypatch,
    ):
        runner = SemgrepRunner()

        monkeypatch.setattr(
            "src.security.semgrep_runner.shutil.which",
            lambda executable: None,
        )

        report = runner.analyze_to_report(
            "print('hello')",
            file_name="example.py",
            language="python",
        )

        assert report.analyzer_name == "semgrep"
        assert report.findings == ()
        assert report.decision == SecurityDecision.PASS
        assert report.metadata["available"] is False

    def test_report_rejects_high_severity_finding(
        self,
        monkeypatch,
    ):
        runner = SemgrepRunner()

        payload = {
            "results": [
                {
                    "check_id": "dangerous-rule",
                    "path": "example.py",
                    "start": {"line": 1},
                    "end": {"line": 1},
                    "extra": {
                        "message": "Dangerous operation",
                        "severity": "ERROR",
                    },
                }
            ]
        }

        monkeypatch.setattr(
            "src.security.semgrep_runner.shutil.which",
            lambda executable: "semgrep",
        )

        monkeypatch.setattr(
            "src.security.semgrep_runner.subprocess.run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=args[0],
                returncode=1,
                stdout=json.dumps(payload),
                stderr="",
            ),
        )

        report = runner.analyze_to_report(
            "dangerous_code()",
            file_name="example.py",
            language="python",
        )

        assert report.decision == SecurityDecision.REJECT
        assert report.finding_count == 1

    def test_report_flags_medium_finding(
        self,
        monkeypatch,
    ):
        runner = SemgrepRunner()

        payload = {
            "results": [
                {
                    "check_id": "medium-rule",
                    "path": "example.py",
                    "start": {"line": 1},
                    "end": {"line": 1},
                    "extra": {
                        "message": "Potential issue",
                        "severity": "WARNING",
                    },
                }
            ]
        }

        monkeypatch.setattr(
            "src.security.semgrep_runner.shutil.which",
            lambda executable: "semgrep",
        )

        monkeypatch.setattr(
            "src.security.semgrep_runner.subprocess.run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=args[0],
                returncode=1,
                stdout=json.dumps(payload),
                stderr="",
            ),
        )

        report = runner.analyze_to_report(
            "potential_code()",
            file_name="example.py",
            language="python",
        )

        assert report.decision == SecurityDecision.FLAG
        assert report.finding_count == 1


class TestHelpers:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("critical", SecuritySeverity.CRITICAL),
            ("ERROR", SecuritySeverity.HIGH),
            ("high", SecuritySeverity.HIGH),
            ("warning", SecuritySeverity.MEDIUM),
            ("medium", SecuritySeverity.MEDIUM),
            ("low", SecuritySeverity.LOW),
            ("info", SecuritySeverity.INFO),
            ("unknown", SecuritySeverity.MEDIUM),
        ],
    )
    def test_severity_mapping(self, value, expected):
        assert SemgrepRunner._map_severity(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("very high", 0.95),
            ("high", 0.85),
            ("medium", 0.70),
            ("low", 0.50),
            ("very low", 0.25),
            ("unknown", 0.70),
        ],
    )
    def test_confidence_mapping(self, value, expected):
        assert SemgrepRunner._map_confidence(value) == expected

    def test_numeric_confidence(self):
        assert SemgrepRunner._map_confidence(0.42) == 0.42

    def test_numeric_confidence_is_clamped(self):
        assert SemgrepRunner._map_confidence(-1) == 0.0
        assert SemgrepRunner._map_confidence(5) == 1.0

    def test_optional_int(self):
        assert SemgrepRunner._optional_int(5) == 5
        assert SemgrepRunner._optional_int("7") == 7
        assert SemgrepRunner._optional_int(None) is None
        assert SemgrepRunner._optional_int("invalid") is None

    def test_aggregate_decision_pass(self):
        assert (
            SemgrepRunner._aggregate_decision(())
            == SecurityDecision.PASS
        )

    def test_aggregate_decision_flag(self):
        finding = make_finding(SecurityDecision.FLAG)

        assert (
            SemgrepRunner._aggregate_decision((finding,))
            == SecurityDecision.FLAG
        )

    def test_aggregate_decision_reject(self):
        finding = make_finding(SecurityDecision.REJECT)

        assert (
            SemgrepRunner._aggregate_decision((finding,))
            == SecurityDecision.REJECT
        )