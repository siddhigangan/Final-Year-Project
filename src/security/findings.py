"""Standardized security findings for SecureCodeRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.models import SecurityDecision, SecuritySeverity


class SecurityFindingError(ValueError):
    """Base exception for security-finding errors."""


class SecurityFindingInputError(SecurityFindingError):
    """Raised when security-finding input is invalid."""


@dataclass(frozen=True)
class SecurityLocation:
    """Location of a security finding within generated code."""

    file_name: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    column_start: int | None = None
    column_end: int | None = None

    def __post_init__(self) -> None:
        """Validate source-location information."""

        if self.line_start is not None and self.line_start <= 0:
            raise SecurityFindingInputError(
                "line_start must be positive when provided."
            )

        if self.line_end is not None and self.line_end <= 0:
            raise SecurityFindingInputError(
                "line_end must be positive when provided."
            )

        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise SecurityFindingInputError(
                "line_end cannot be smaller than line_start."
            )

        if self.column_start is not None and self.column_start < 0:
            raise SecurityFindingInputError(
                "column_start cannot be negative."
            )

        if self.column_end is not None and self.column_end < 0:
            raise SecurityFindingInputError(
                "column_end cannot be negative."
            )

    def as_dict(self) -> dict[str, Any]:
        """Return the location as a serializable dictionary."""

        return {
            "file_name": self.file_name,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "column_start": self.column_start,
            "column_end": self.column_end,
        }


@dataclass(frozen=True)
class SecurityFinding:
    """A standardized security finding."""

    finding_id: str
    category: str
    rule_id: str
    title: str
    description: str
    severity: SecuritySeverity
    confidence: float
    decision: SecurityDecision
    location: SecurityLocation | None = None
    evidence: str = ""
    remediation: str = ""
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the finding."""

        required_strings = {
            "finding_id": self.finding_id,
            "category": self.category,
            "rule_id": self.rule_id,
            "title": self.title,
            "description": self.description,
            "source": self.source,
        }

        for name, value in required_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise SecurityFindingInputError(
                    f"{name} must be a non-empty string."
                )

        if not isinstance(self.severity, SecuritySeverity):
            raise SecurityFindingInputError(
                "severity must be a SecuritySeverity value."
            )

        if not isinstance(self.decision, SecurityDecision):
            raise SecurityFindingInputError(
                "decision must be a SecurityDecision value."
            )

        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise SecurityFindingInputError(
                "confidence must be a number between 0.0 and 1.0."
            )

        if not isinstance(self.evidence, str):
            raise SecurityFindingInputError(
                "evidence must be a string."
            )

        if not isinstance(self.remediation, str):
            raise SecurityFindingInputError(
                "remediation must be a string."
            )

        if not isinstance(self.metadata, dict):
            raise SecurityFindingInputError(
                "metadata must be a dictionary."
            )

    @property
    def is_security_issue(self) -> bool:
        """Return whether the finding represents a detected issue."""

        return self.decision in {
            SecurityDecision.REJECT,
            SecurityDecision.FLAG,
        }

    @property
    def is_blocking(self) -> bool:
        """Return whether the finding should block output."""

        return self.decision == SecurityDecision.REJECT

    def as_dict(self) -> dict[str, Any]:
        """Return the finding as a serializable dictionary."""

        return {
            "finding_id": self.finding_id,
            "category": self.category,
            "rule_id": self.rule_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": float(self.confidence),
            "decision": self.decision.value,
            "location": (
                self.location.as_dict()
                if self.location is not None
                else None
            ),
            "evidence": self.evidence,
            "remediation": self.remediation,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SecurityReport:
    """Collection of security findings for one analyzed output."""

    findings: tuple[SecurityFinding, ...] = ()
    decision: SecurityDecision = SecurityDecision.PASS
    analyzer_name: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the security report."""

        if not isinstance(self.findings, tuple):
            raise SecurityFindingInputError(
                "findings must be a tuple of SecurityFinding objects."
            )

        if any(
            not isinstance(finding, SecurityFinding)
            for finding in self.findings
        ):
            raise SecurityFindingInputError(
                "findings must contain only SecurityFinding objects."
            )

        if not isinstance(self.decision, SecurityDecision):
            raise SecurityFindingInputError(
                "decision must be a SecurityDecision value."
            )

        if not isinstance(self.analyzer_name, str):
            raise SecurityFindingInputError(
                "analyzer_name must be a string."
            )

        if not self.analyzer_name.strip():
            raise SecurityFindingInputError(
                "analyzer_name cannot be empty."
            )

        if not isinstance(self.metadata, dict):
            raise SecurityFindingInputError(
                "metadata must be a dictionary."
            )

    @property
    def finding_count(self) -> int:
        """Return the number of findings."""

        return len(self.findings)

    @property
    def blocking_findings(self) -> tuple[SecurityFinding, ...]:
        """Return findings that require blocking."""

        return tuple(
            finding
            for finding in self.findings
            if finding.is_blocking
        )

    @property
    def warning_findings(self) -> tuple[SecurityFinding, ...]:
        """Return findings that produce warnings."""

        return tuple(
            finding
            for finding in self.findings
            if finding.decision == SecurityDecision.FLAG
        )

    @property
    def highest_severity(self) -> SecuritySeverity | None:
        """Return the highest severity present in the report."""

        if not self.findings:
            return None

        severity_order = {
            SecuritySeverity.INFO: 0,
            SecuritySeverity.LOW: 1,
            SecuritySeverity.MEDIUM: 2,
            SecuritySeverity.HIGH: 3,
            SecuritySeverity.CRITICAL: 4,
        }

        return max(
            self.findings,
            key=lambda finding: severity_order[finding.severity],
        ).severity

    def as_dict(self) -> dict[str, Any]:
        """Return the report as a serializable dictionary."""

        return {
            "findings": [
                finding.as_dict()
                for finding in self.findings
            ],
            "finding_count": self.finding_count,
            "blocking_findings": len(self.blocking_findings),
            "warning_findings": len(self.warning_findings),
            "highest_severity": (
                self.highest_severity.value
                if self.highest_severity is not None
                else None
            ),
            "decision": self.decision.value,
            "analyzer_name": self.analyzer_name,
            "metadata": dict(self.metadata),
        }


def combine_security_reports(
    reports: list[SecurityReport],
    analyzer_name: str = "combined",
) -> SecurityReport:
    """Combine multiple analyzer reports into one report."""

    if not isinstance(reports, list):
        raise SecurityFindingInputError(
            "reports must be a list."
        )

    if any(
        not isinstance(report, SecurityReport)
        for report in reports
    ):
        raise SecurityFindingInputError(
            "reports must contain only SecurityReport objects."
        )

    findings = tuple(
        finding
        for report in reports
        for finding in report.findings
    )

    if any(
        finding.decision == SecurityDecision.REJECT
        for finding in findings
    ):
        decision = SecurityDecision.REJECT
    elif any(
        finding.decision == SecurityDecision.FLAG
        for finding in findings
    ):
        decision = SecurityDecision.FLAG
    else:
        decision = SecurityDecision.PASS

    return SecurityReport(
        findings=findings,
        decision=decision,
        analyzer_name=analyzer_name,
        metadata={
            "source_reports": len(reports),
        },
    )