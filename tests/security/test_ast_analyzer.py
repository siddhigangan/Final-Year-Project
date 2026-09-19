"""Tests for Python AST security analysis."""

from __future__ import annotations

import pytest

from src.models import SecurityDecision, SecuritySeverity
from src.security.ast_analyzer import (
    ASTAnalyzerInputError,
    ASTParseError,
    ASTSecurityAnalyzer,
    SecurityRule,
    analyze_python_code,
)


class TestASTSecurityAnalyzer:
    """Tests for ASTSecurityAnalyzer."""

    def test_clean_code_returns_pass(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(
            """
def add(a, b):
    return a + b
"""
        )

        assert report.findings == ()
        assert report.decision == SecurityDecision.PASS
        assert report.analyzer_name == "python_ast"

    def test_eval_is_detected(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze("result = eval(user_input)")

        assert report.finding_count == 1
        assert report.findings[0].rule_id == "PY-EVAL-001"
        assert report.findings[0].severity == SecuritySeverity.HIGH
        assert report.decision == SecurityDecision.REJECT

    def test_exec_is_detected(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze("exec(user_code)")

        assert report.finding_count == 1
        assert report.findings[0].rule_id == "PY-EXEC-001"
        assert report.findings[0].severity == SecuritySeverity.CRITICAL
        assert report.decision == SecurityDecision.REJECT

    def test_os_system_is_detected(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze("os.system(command)")

        assert report.finding_count == 1
        assert report.findings[0].rule_id == "PY-OS-001"
        assert report.findings[0].severity == SecuritySeverity.HIGH
        assert report.decision == SecurityDecision.REJECT

    @pytest.mark.parametrize(
        "call",
        [
            "subprocess.run(command)",
            "subprocess.call(command)",
            "subprocess.Popen(command)",
            "subprocess.check_call(command)",
            "subprocess.check_output(command)",
        ],
    )
    def test_subprocess_calls_are_detected(
        self,
        call: str,
    ) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(call)

        assert report.finding_count == 1
        assert report.findings[0].rule_id == "PY-SUBPROCESS-001"
        assert report.findings[0].severity == SecuritySeverity.MEDIUM
        assert report.decision == SecurityDecision.REJECT

    @pytest.mark.parametrize(
        "call",
        [
            "pickle.load(file_obj)",
            "pickle.loads(data)",
        ],
    )
    def test_pickle_deserialization_is_detected(
        self,
        call: str,
    ) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(call)

        assert report.finding_count == 1
        assert report.findings[0].rule_id == "PY-PICKLE-001"
        assert report.findings[0].severity == SecuritySeverity.HIGH
        assert report.decision == SecurityDecision.REJECT

    def test_yaml_load_is_detected(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze("data = yaml.load(content)")

        assert report.finding_count == 1
        assert report.findings[0].rule_id == "PY-YAML-001"
        assert report.findings[0].severity == SecuritySeverity.HIGH
        assert report.decision == SecurityDecision.REJECT

    @pytest.mark.parametrize(
        "call",
        [
            "hashlib.md5(data)",
            "hashlib.sha1(data)",
        ],
    )
    def test_weak_hashes_are_detected(
        self,
        call: str,
    ) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(call)

        assert report.finding_count == 1
        assert report.findings[0].rule_id == "PY-CRYPTO-001"
        assert report.findings[0].severity == SecuritySeverity.MEDIUM
        assert report.decision == SecurityDecision.REJECT

    @pytest.mark.parametrize(
        "code",
        [
            'password = "super-secret-password"',
            'api_key = "123456789"',
            'secret = "my-secret"',
            'token = "abc123"',
            'private_key = "PRIVATE_KEY_VALUE"',
            'client_secret = "CLIENT_SECRET_VALUE"',
        ],
    )
    def test_hardcoded_secrets_are_detected(
        self,
        code: str,
    ) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(code)

        assert report.finding_count == 1
        assert report.findings[0].rule_id == "PY-SECRET-001"
        assert report.findings[0].severity == SecuritySeverity.HIGH
        assert report.decision == SecurityDecision.REJECT

    @pytest.mark.parametrize(
        "code",
        [
            'query = "SELECT * FROM users WHERE id=" + user_id',
            'query = f"SELECT * FROM users WHERE id={user_id}"',
            'query = "SELECT * FROM users WHERE id=%s" % user_id',
        ],
    )
    def test_sql_construction_is_detected(
        self,
        code: str,
    ) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(code)

        assert report.finding_count >= 1
        assert any(
            finding.rule_id == "PY-SQL-001"
            for finding in report.findings
        )
        assert any(
            finding.severity == SecuritySeverity.HIGH
            for finding in report.findings
        )
        assert report.decision == SecurityDecision.REJECT

    def test_sql_in_function_call_is_detected(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(
            'cursor.execute("SELECT * FROM users WHERE id=" + user_id)'
        )

        assert report.finding_count >= 1
        assert any(
            finding.rule_id == "PY-SQL-001"
            for finding in report.findings
        )
        assert report.decision == SecurityDecision.REJECT

    def test_sql_f_string_in_function_call_is_detected(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(
            'cursor.execute(f"SELECT * FROM users WHERE id={user_id}")'
        )

        assert report.finding_count >= 1
        assert any(
            finding.rule_id == "PY-SQL-001"
            for finding in report.findings
        )
        assert report.decision == SecurityDecision.REJECT

    def test_parameterized_sql_is_not_flagged(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(
            'cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))'
        )

        assert not any(
            finding.rule_id == "PY-SQL-001"
            for finding in report.findings
        )

    def test_safe_hash_is_not_flagged(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze("hashlib.sha256(data).hexdigest()")

        assert not any(
            finding.rule_id == "PY-CRYPTO-001"
            for finding in report.findings
        )

    def test_safe_yaml_loader_is_not_flagged(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze("data = yaml.safe_load(content)")

        assert not any(
            finding.rule_id == "PY-YAML-001"
            for finding in report.findings
        )

    def test_environment_secret_is_not_flagged_as_hardcoded_secret(
        self,
    ) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(
            'api_key = os.getenv("API_KEY")'
        )

        assert not any(
            finding.rule_id == "PY-SECRET-001"
            for finding in report.findings
        )

    def test_annotated_hardcoded_secret_is_detected(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(
            'password: str = "hardcoded-password"'
        )

        assert report.finding_count == 1
        assert report.findings[0].rule_id == "PY-SECRET-001"
        assert report.decision == SecurityDecision.REJECT

    def test_multiple_vulnerabilities_are_reported(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        code = """
import os
import pickle

result = eval(user_input)
os.system(command)
data = pickle.loads(payload)
password = "super-secret"
"""

        report = analyzer.analyze(code)

        rule_ids = {
            finding.rule_id
            for finding in report.findings
        }

        assert "PY-EVAL-001" in rule_ids
        assert "PY-OS-001" in rule_ids
        assert "PY-PICKLE-001" in rule_ids
        assert "PY-SECRET-001" in rule_ids

        assert report.finding_count == 4
        assert report.decision == SecurityDecision.REJECT

        # The configured pickle rule is HIGH, while the EXEC rule
        # is the configured CRITICAL rule.
        assert report.highest_severity == SecuritySeverity.HIGH

    def test_finding_contains_source_location(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(
            "result = eval(user_input)",
            file_name="example.py",
        )

        finding = report.findings[0]

        assert finding.location.file_name == "example.py"
        assert finding.location.line_start == 1
        assert finding.location.line_end == 1

    def test_finding_contains_remediation(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze("result = eval(user_input)")

        finding = report.findings[0]

        assert finding.remediation
        assert "eval" in finding.remediation.lower()

    def test_report_metadata_contains_language(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze("x = 1")

        assert report.metadata["language"] == "python"
        assert report.metadata["rule_count"] == 9

    def test_custom_analyzer_name(self) -> None:
        analyzer = ASTSecurityAnalyzer(
            analyzer_name="custom_python_analyzer",
        )

        report = analyzer.analyze("x = 1")

        assert report.analyzer_name == "custom_python_analyzer"

    def test_custom_rules_are_supported(self) -> None:
        custom_rule = SecurityRule(
            rule_id="CUSTOM-001",
            category="custom",
            title="Custom rule",
            description="Custom security rule.",
            severity=SecuritySeverity.LOW,
            remediation="Fix the issue.",
        )

        analyzer = ASTSecurityAnalyzer(
            rules=(custom_rule,),
        )

        report = analyzer.analyze("x = 1")

        assert report.metadata["rule_count"] == 1
        assert report.findings == ()

    def test_non_string_input_raises_input_error(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        with pytest.raises(ASTAnalyzerInputError):
            analyzer.analyze(None)  # type: ignore[arg-type]

    def test_invalid_python_raises_parse_error(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        with pytest.raises(ASTParseError):
            analyzer.analyze(
                "def broken("
            )

    def test_invalid_rules_type_raises_input_error(self) -> None:
        with pytest.raises(ASTAnalyzerInputError):
            ASTSecurityAnalyzer(
                rules=[]  # type: ignore[arg-type]
            )

    def test_invalid_rule_member_raises_input_error(self) -> None:
        with pytest.raises(ASTAnalyzerInputError):
            ASTSecurityAnalyzer(
                rules=("not-a-security-rule",)  # type: ignore[arg-type]
            )

    def test_empty_analyzer_name_raises_input_error(self) -> None:
        with pytest.raises(ASTAnalyzerInputError):
            ASTSecurityAnalyzer(analyzer_name="")

    def test_convenience_function_works(self) -> None:
        report = analyze_python_code(
            "result = eval(user_input)",
            file_name="generated.py",
        )

        assert report.finding_count == 1
        assert report.findings[0].rule_id == "PY-EVAL-001"
        assert report.findings[0].location.file_name == "generated.py"

    def test_report_as_dict_contains_findings(self) -> None:
        analyzer = ASTSecurityAnalyzer()

        report = analyzer.analyze(
            "result = eval(user_input)"
        )

        serialized = report.as_dict()

        assert serialized["decision"] == "reject"
        assert len(serialized["findings"]) == 1
        assert serialized["findings"][0]["rule_id"] == "PY-EVAL-001"