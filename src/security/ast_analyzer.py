"""Static security analysis for generated Python code."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from src.models import SecurityDecision, SecuritySeverity
from src.security.findings import (
    SecurityFinding,
    SecurityLocation,
    SecurityReport,
)


class ASTAnalyzerError(ValueError):
    """Base exception for AST analyzer errors."""


class ASTAnalyzerInputError(ASTAnalyzerError):
    """Raised when analyzer input is invalid."""


class ASTParseError(ASTAnalyzerError):
    """Raised when generated code cannot be parsed."""


@dataclass(frozen=True)
class SecurityRule:
    """Definition of one AST security rule."""

    rule_id: str
    category: str
    title: str
    description: str
    severity: SecuritySeverity
    remediation: str


DEFAULT_RULES: tuple[SecurityRule, ...] = (
    SecurityRule(
        rule_id="PY-EVAL-001",
        category="code_execution",
        title="Dynamic eval execution",
        description=(
            "The generated code invokes eval(), which can execute "
            "attacker-controlled Python expressions."
        ),
        severity=SecuritySeverity.HIGH,
        remediation=(
            "Avoid eval(). Use explicit parsing or a constrained "
            "operation mapping instead."
        ),
    ),
    SecurityRule(
        rule_id="PY-EXEC-001",
        category="code_execution",
        title="Dynamic exec execution",
        description=(
            "The generated code invokes exec(), allowing arbitrary "
            "Python code execution."
        ),
        severity=SecuritySeverity.CRITICAL,
        remediation=(
            "Avoid exec(). Replace dynamic code execution with "
            "explicit program logic."
        ),
    ),
    SecurityRule(
        rule_id="PY-OS-001",
        category="command_execution",
        title="Operating-system command execution",
        description=(
            "The generated code uses os.system(), which executes "
            "a command through the operating-system shell."
        ),
        severity=SecuritySeverity.HIGH,
        remediation=(
            "Avoid os.system(). Use a constrained subprocess API "
            "with validated arguments when process execution is required."
        ),
    ),
    SecurityRule(
        rule_id="PY-SUBPROCESS-001",
        category="command_execution",
        title="Potentially unsafe subprocess execution",
        description=(
            "The generated code invokes subprocess execution and "
            "requires validation of command construction and arguments."
        ),
        severity=SecuritySeverity.MEDIUM,
        remediation=(
            "Prefer fixed command arguments and avoid shell execution. "
            "Validate all external input before process execution."
        ),
    ),
    SecurityRule(
        rule_id="PY-PICKLE-001",
        category="unsafe_deserialization",
        title="Unsafe pickle deserialization",
        description=(
            "The generated code uses pickle deserialization, which "
            "can execute arbitrary code when processing untrusted data."
        ),
        severity=SecuritySeverity.HIGH,
        remediation=(
            "Do not deserialize untrusted data with pickle. "
            "Use a safe data serialization format."
        ),
    ),
    SecurityRule(
        rule_id="PY-YAML-001",
        category="unsafe_deserialization",
        title="Potentially unsafe YAML loading",
        description=(
            "The generated code uses yaml.load() without an explicitly "
            "safe loader."
        ),
        severity=SecuritySeverity.HIGH,
        remediation=(
            "Use yaml.safe_load() or explicitly configure a safe loader."
        ),
    ),
    SecurityRule(
        rule_id="PY-CRYPTO-001",
        category="weak_cryptography",
        title="Weak cryptographic hash",
        description=(
            "The generated code uses MD5 or SHA-1, which are unsuitable "
            "for security-sensitive hashing."
        ),
        severity=SecuritySeverity.MEDIUM,
        remediation=(
            "Use SHA-256 or a modern password-hashing algorithm "
            "appropriate to the security requirement."
        ),
    ),
    SecurityRule(
        rule_id="PY-SQL-001",
        category="sql_injection",
        title="Potential SQL injection",
        description=(
            "The generated code constructs a SQL statement using "
            "string concatenation or interpolation."
        ),
        severity=SecuritySeverity.HIGH,
        remediation=(
            "Use parameterized queries instead of constructing SQL "
            "statements through string concatenation or interpolation."
        ),
    ),
    SecurityRule(
        rule_id="PY-SECRET-001",
        category="hardcoded_secret",
        title="Potential hard-coded secret",
        description=(
            "The generated code contains a string assignment that "
            "appears to contain a credential or secret."
        ),
        severity=SecuritySeverity.HIGH,
        remediation=(
            "Load secrets from a secure secret manager or environment "
            "configuration rather than storing them in source code."
        ),
    ),
)


_SECRET_NAME_PATTERN = re.compile(
    r"(password|passwd|secret|api[_-]?key|token|private[_-]?key|"
    r"access[_-]?key|client[_-]?secret)",
    re.IGNORECASE,
)

_SQL_PATTERN = re.compile(
    r"\b(select|insert|update|delete|drop|alter|create)\b",
    re.IGNORECASE,
)


class ASTSecurityAnalyzer:
    """Analyze generated Python code for deterministic security patterns."""

    def __init__(
        self,
        rules: tuple[SecurityRule, ...] = DEFAULT_RULES,
        analyzer_name: str = "python_ast",
    ) -> None:
        if not isinstance(rules, tuple):
            raise ASTAnalyzerInputError(
                "rules must be a tuple of SecurityRule objects."
            )

        if any(
            not isinstance(rule, SecurityRule)
            for rule in rules
        ):
            raise ASTAnalyzerInputError(
                "rules must contain only SecurityRule objects."
            )

        if not isinstance(analyzer_name, str) or not analyzer_name.strip():
            raise ASTAnalyzerInputError(
                "analyzer_name must be a non-empty string."
            )

        self.rules = rules
        self.analyzer_name = analyzer_name

        self._rules_by_id = {
            rule.rule_id: rule
            for rule in rules
        }

    def analyze(
        self,
        code: str,
        file_name: str | None = None,
    ) -> SecurityReport:
        """Analyze Python source code and return a security report."""

        if not isinstance(code, str):
            raise ASTAnalyzerInputError("code must be a string.")

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ASTParseError(
                "Generated Python code could not be parsed."
            ) from exc

        findings: list[SecurityFinding] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                findings.extend(
                    self._analyze_call(
                        node=node,
                        code=code,
                        file_name=file_name,
                    )
                )

            elif isinstance(node, ast.Assign):
                findings.extend(
                    self._analyze_assignment(
                        node=node,
                        code=code,
                        file_name=file_name,
                    )
                )

            elif isinstance(node, ast.AnnAssign):
                findings.extend(
                    self._analyze_annotated_assignment(
                        node=node,
                        code=code,
                        file_name=file_name,
                    )
                )

        return SecurityReport(
            findings=tuple(findings),
            decision=self._overall_decision(findings),
            analyzer_name=self.analyzer_name,
            metadata={
                "language": "python",
                "file_name": file_name,
                "rule_count": len(self.rules),
            },
        )

    def _analyze_call(
        self,
        node: ast.Call,
        code: str,
        file_name: str | None,
    ) -> list[SecurityFinding]:
        """Analyze a function call."""

        call_name = self._call_name(node)

        if call_name == "eval":
            return [
                self._finding(
                    rule_id="PY-EVAL-001",
                    node=node,
                    file_name=file_name,
                    evidence="eval(...)",
                )
            ]

        if call_name == "exec":
            return [
                self._finding(
                    rule_id="PY-EXEC-001",
                    node=node,
                    file_name=file_name,
                    evidence="exec(...)",
                )
            ]

        if call_name == "os.system":
            return [
                self._finding(
                    rule_id="PY-OS-001",
                    node=node,
                    file_name=file_name,
                    evidence="os.system(...)",
                )
            ]

        if call_name in {
            "subprocess.run",
            "subprocess.call",
            "subprocess.Popen",
            "subprocess.check_call",
            "subprocess.check_output",
        }:
            return [
                self._finding(
                    rule_id="PY-SUBPROCESS-001",
                    node=node,
                    file_name=file_name,
                    evidence=call_name,
                )
            ]

        if call_name in {
            "pickle.load",
            "pickle.loads",
        }:
            return [
                self._finding(
                    rule_id="PY-PICKLE-001",
                    node=node,
                    file_name=file_name,
                    evidence=call_name,
                )
            ]

        if call_name == "yaml.load":
            return [
                self._finding(
                    rule_id="PY-YAML-001",
                    node=node,
                    file_name=file_name,
                    evidence="yaml.load(...)",
                )
            ]

        if call_name in {
            "hashlib.md5",
            "hashlib.sha1",
        }:
            return [
                self._finding(
                    rule_id="PY-CRYPTO-001",
                    node=node,
                    file_name=file_name,
                    evidence=call_name,
                )
            ]

        if self._looks_like_sql_call(node, code):
            return [
                self._finding(
                    rule_id="PY-SQL-001",
                    node=node,
                    file_name=file_name,
                    evidence=self._source_segment(code, node),
                )
            ]

        return []

    def _analyze_assignment(
        self,
        node: ast.Assign,
        code: str,
        file_name: str | None,
    ) -> list[SecurityFinding]:
        """Analyze assignments for secrets and SQL construction."""

        findings: list[SecurityFinding] = []

        value = self._string_value(node.value)

        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue

            if (
                _SECRET_NAME_PATTERN.search(target.id)
                and value is not None
            ):
                findings.append(
                    self._finding(
                        rule_id="PY-SECRET-001",
                        node=node,
                        file_name=file_name,
                        evidence=f"{target.id}=<redacted>",
                    )
                )

        if self._looks_like_sql_assignment(node.value, code):
            findings.append(
                self._finding(
                    rule_id="PY-SQL-001",
                    node=node,
                    file_name=file_name,
                    evidence=self._source_segment(code, node),
                )
            )

        return findings

    def _analyze_annotated_assignment(
        self,
        node: ast.AnnAssign,
        code: str,
        file_name: str | None,
    ) -> list[SecurityFinding]:
        """Analyze annotated assignments for hard-coded secrets."""

        del code

        if not isinstance(node.target, ast.Name):
            return []

        if not _SECRET_NAME_PATTERN.search(node.target.id):
            return []

        value = self._string_value(node.value) if node.value else None

        if value is None:
            return []

        return [
            self._finding(
                rule_id="PY-SECRET-001",
                node=node,
                file_name=file_name,
                evidence=f"{node.target.id}=<redacted>",
            )
        ]

    def _looks_like_sql_assignment(
        self,
        node: ast.AST,
        code: str,
    ) -> bool:
        """
        Detect SQL construction in an assignment expression.

        SQL must contain a recognized SQL keyword and must be combined
        with dynamic content through concatenation, interpolation,
        or percent formatting.
        """

        source = self._source_segment(code, node)

        if not source or source == "<source unavailable>":
            return False

        if not _SQL_PATTERN.search(source):
            return False

        if isinstance(node, ast.JoinedStr):
            return True

        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Mod),
        ):
            return True

        return self._contains_dynamic_sql_component(node)

    def _looks_like_sql_call(
        self,
        node: ast.Call,
        code: str,
    ) -> bool:
        """Detect unsafe SQL construction passed to a function."""

        if not node.args:
            return False

        first_argument = node.args[0]
        source = self._source_segment(code, first_argument)

        if not source or source == "<source unavailable>":
            return False

        if not _SQL_PATTERN.search(source):
            return False

        if isinstance(first_argument, ast.JoinedStr):
            return True

        if isinstance(first_argument, ast.BinOp) and isinstance(
            first_argument.op,
            (ast.Add, ast.Mod),
        ):
            return True

        value = self._string_value(first_argument)

        if value is not None:
            # A constant SQL string followed by a separate parameters
            # argument represents a parameterized query.
            return False

        return self._contains_dynamic_sql_component(first_argument)

    def _contains_dynamic_sql_component(
        self,
        node: ast.AST,
    ) -> bool:
        """Determine whether an expression contains dynamic content."""

        if isinstance(node, ast.Name):
            return True

        if isinstance(node, ast.JoinedStr):
            return True

        if isinstance(node, ast.FormattedValue):
            return True

        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Mod),
        ):
            return True

        return any(
            self._contains_dynamic_sql_component(child)
            for child in ast.iter_child_nodes(node)
        )

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        """Return a normalized function name for a call."""

        function = node.func

        if isinstance(function, ast.Name):
            return function.id

        if isinstance(function, ast.Attribute):
            parts: list[str] = []
            current: ast.AST | None = function

            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                parts.append(current.id)

            return ".".join(reversed(parts))

        return ""

    @staticmethod
    def _string_value(
        node: ast.AST | None,
    ) -> str | None:
        """Extract a constant string value when available."""

        if isinstance(node, ast.Constant) and isinstance(
            node.value,
            str,
        ):
            return node.value

        return None

    @staticmethod
    def _source_segment(
        code: str,
        node: ast.AST,
    ) -> str:
        """Return source text for an AST node."""

        segment = ast.get_source_segment(code, node)

        if segment is None:
            return "<source unavailable>"

        return segment[:500]

    def _finding(
        self,
        rule_id: str,
        node: ast.AST,
        file_name: str | None,
        evidence: str,
    ) -> SecurityFinding:
        """Create a standardized security finding."""

        rule = self._rules_by_id.get(rule_id)

        if rule is None:
            raise ASTAnalyzerError(
                f"Security rule '{rule_id}' is not configured."
            )

        return SecurityFinding(
            finding_id=(
                f"{rule.rule_id}:"
                f"{getattr(node, 'lineno', 0)}:"
                f"{getattr(node, 'col_offset', 0)}"
            ),
            category=rule.category,
            rule_id=rule.rule_id,
            title=rule.title,
            description=rule.description,
            severity=rule.severity,
            confidence=0.90,
            decision=SecurityDecision.REJECT,
            location=SecurityLocation(
                file_name=file_name,
                line_start=getattr(node, "lineno", None),
                line_end=getattr(
                    node,
                    "end_lineno",
                    getattr(node, "lineno", None),
                ),
                column_start=getattr(node, "col_offset", None),
                column_end=getattr(node, "end_col_offset", None),
            ),
            evidence=evidence,
            remediation=rule.remediation,
            source=self.analyzer_name,
        )

    @staticmethod
    def _overall_decision(
        findings: list[SecurityFinding],
    ) -> SecurityDecision:
        """Determine the overall report decision."""

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


def analyze_python_code(
    code: str,
    file_name: str | None = None,
) -> SecurityReport:
    """Convenience function for Python security analysis."""

    analyzer = ASTSecurityAnalyzer()

    return analyzer.analyze(
        code=code,
        file_name=file_name,
    )