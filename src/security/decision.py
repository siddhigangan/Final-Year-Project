from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.models import SecurityDecision, SecuritySeverity
from src.security.findings import SecurityFinding, SecurityReport


class SecurityDecisionError(ValueError):
    """Raised when security decision input is invalid."""


@dataclass(frozen=True)
class SecurityDecisionPolicy:
    """Configurable policy for converting findings into a security decision."""

    reject_severities: frozenset[SecuritySeverity] = frozenset(
        {
            SecuritySeverity.CRITICAL,
            SecuritySeverity.HIGH,
        }
    )
    flag_severities: frozenset[SecuritySeverity] = frozenset(
        {
            SecuritySeverity.MEDIUM,
            SecuritySeverity.LOW,
            SecuritySeverity.INFO,
        }
    )
    reject_on_blocking_finding: bool = True

    def __post_init__(self) -> None:
        if self.reject_severities & self.flag_severities:
            raise SecurityDecisionError(
                "reject_severities and flag_severities must not overlap"
            )


class SecurityDecisionEngine:
    """Convert security findings into deterministic security decisions."""

    def __init__(
        self,
        policy: SecurityDecisionPolicy | None = None,
    ) -> None:
        self._policy = policy or SecurityDecisionPolicy()

    @property
    def policy(self) -> SecurityDecisionPolicy:
        """Return the active decision policy."""
        return self._policy

    def decide_finding(
        self,
        finding: SecurityFinding,
    ) -> SecurityDecision:
        """Return the decision associated with one security finding."""
        if not isinstance(finding, SecurityFinding):
            raise SecurityDecisionError(
                "finding must be a SecurityFinding instance"
            )

        if (
            self._policy.reject_on_blocking_finding
            and finding.is_blocking
        ):
            return SecurityDecision.REJECT

        if finding.severity in self._policy.reject_severities:
            return SecurityDecision.REJECT

        if finding.severity in self._policy.flag_severities:
            return SecurityDecision.FLAG

        return SecurityDecision.PASS

    def decide_findings(
        self,
        findings: Iterable[SecurityFinding],
    ) -> SecurityDecision:
        """Aggregate multiple findings into one security decision."""
        if findings is None:
            raise SecurityDecisionError("findings cannot be None")

        highest_decision = SecurityDecision.PASS

        for finding in findings:
            decision = self.decide_finding(finding)

            if decision == SecurityDecision.REJECT:
                return SecurityDecision.REJECT

            if (
                decision == SecurityDecision.FLAG
                and highest_decision == SecurityDecision.PASS
            ):
                highest_decision = SecurityDecision.FLAG

        return highest_decision

    def decide_report(
        self,
        report: SecurityReport,
    ) -> SecurityDecision:
        """Evaluate an existing security report."""
        if not isinstance(report, SecurityReport):
            raise SecurityDecisionError(
                "report must be a SecurityReport instance"
            )

        return self.decide_findings(report.findings)

    def evaluate(
        self,
        findings: Iterable[SecurityFinding],
    ) -> SecurityReport:
        """Build a normalized SecurityReport from security findings."""
        if findings is None:
            raise SecurityDecisionError("findings cannot be None")

        normalized_findings = tuple(findings)

        for finding in normalized_findings:
            if not isinstance(finding, SecurityFinding):
                raise SecurityDecisionError(
                    "all findings must be SecurityFinding instances"
                )

        decision = self.decide_findings(normalized_findings)

        return SecurityReport(
            findings=normalized_findings,
            decision=decision,
            analyzer_name="security-decision-engine",
            metadata={
                "policy": {
                    "reject_severities": sorted(
                        severity.value
                        for severity in self._policy.reject_severities
                    ),
                    "flag_severities": sorted(
                        severity.value
                        for severity in self._policy.flag_severities
                    ),
                    "reject_on_blocking_finding": (
                        self._policy.reject_on_blocking_finding
                    ),
                }
            },
        )


__all__ = [
    "SecurityDecisionEngine",
    "SecurityDecisionError",
    "SecurityDecisionPolicy",
]