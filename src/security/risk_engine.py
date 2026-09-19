from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from src.models import SecurityDecision, SecuritySeverity
from src.security.decision import SecurityDecisionEngine
from src.security.findings import SecurityFinding, SecurityReport


class RiskEngineError(RuntimeError):
    """Base exception for risk-engine failures."""


class RiskEngineInputError(RiskEngineError):
    """Raised when risk-engine input is invalid."""


@dataclass(frozen=True)
class RiskWeights:
    """Weights used to calculate normalized security risk."""

    critical: float = 1.00
    high: float = 0.80
    medium: float = 0.50
    low: float = 0.20
    info: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.critical,
            self.high,
            self.medium,
            self.low,
            self.info,
        )

        if any(value < 0.0 for value in values):
            raise RiskEngineInputError(
                "Risk weights cannot be negative"
            )

        if self.critical < self.high:
            raise RiskEngineInputError(
                "Critical risk weight must be >= high risk weight"
            )

        if self.high < self.medium:
            raise RiskEngineInputError(
                "High risk weight must be >= medium risk weight"
            )

        if self.medium < self.low:
            raise RiskEngineInputError(
                "Medium risk weight must be >= low risk weight"
            )

        if self.low < self.info:
            raise RiskEngineInputError(
                "Low risk weight must be >= info risk weight"
            )

    def for_severity(
        self,
        severity: SecuritySeverity,
    ) -> float:
        """Return the configured weight for a severity."""
        mapping = {
            SecuritySeverity.CRITICAL: self.critical,
            SecuritySeverity.HIGH: self.high,
            SecuritySeverity.MEDIUM: self.medium,
            SecuritySeverity.LOW: self.low,
            SecuritySeverity.INFO: self.info,
        }

        return mapping[severity]


@dataclass(frozen=True)
class RiskAssessment:
    """Normalized security-risk assessment."""

    decision: SecurityDecision
    risk_score: float
    finding_count: int
    blocking_finding_count: int
    critical_finding_count: int
    high_finding_count: int
    medium_finding_count: int
    low_finding_count: int
    info_finding_count: int
    analyzer_names: tuple[str, ...] = ()
    findings: tuple[SecurityFinding, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        """Return whether the assessment passed."""
        return self.decision == SecurityDecision.PASS

    @property
    def requires_review(self) -> bool:
        """Return whether the assessment requires review."""
        return self.decision == SecurityDecision.FLAG

    @property
    def must_be_rejected(self) -> bool:
        """Return whether the assessment must be rejected."""
        return self.decision == SecurityDecision.REJECT


class SecurityRiskEngine:
    """Aggregate security findings into a normalized risk assessment."""

    def __init__(
        self,
        decision_engine: SecurityDecisionEngine | None = None,
        weights: RiskWeights | None = None,
        risk_threshold: float = 0.50,
    ) -> None:
        if risk_threshold < 0.0 or risk_threshold > 1.0:
            raise RiskEngineInputError(
                "risk_threshold must be between 0.0 and 1.0"
            )

        self._decision_engine = (
            decision_engine or SecurityDecisionEngine()
        )
        self._weights = weights or RiskWeights()
        self._risk_threshold = risk_threshold

    @property
    def decision_engine(self) -> SecurityDecisionEngine:
        """Return the configured decision engine."""
        return self._decision_engine

    @property
    def weights(self) -> RiskWeights:
        """Return the configured risk weights."""
        return self._weights

    @property
    def risk_threshold(self) -> float:
        """Return the configured risk threshold."""
        return self._risk_threshold

    def assess_findings(
        self,
        findings: Iterable[SecurityFinding],
        *,
        analyzer_names: Iterable[str] = (),
        metadata: dict[str, object] | None = None,
    ) -> RiskAssessment:
        """Assess a collection of security findings."""
        normalized_findings = tuple(findings)

        if any(
            not isinstance(finding, SecurityFinding)
            for finding in normalized_findings
        ):
            raise RiskEngineInputError(
                "All findings must be SecurityFinding instances"
            )

        normalized_analyzers = tuple(
            dict.fromkeys(
                name.strip()
                for name in analyzer_names
                if isinstance(name, str) and name.strip()
            )
        )

        decision = self._decision_engine.decide_findings(
            normalized_findings
        )

        risk_score = self._calculate_risk_score(
            normalized_findings
        )

        if (
            decision == SecurityDecision.PASS
            and risk_score >= self._risk_threshold
            and normalized_findings
        ):
            decision = SecurityDecision.FLAG

        return RiskAssessment(
            decision=decision,
            risk_score=risk_score,
            finding_count=len(normalized_findings),
            blocking_finding_count=sum(
                finding.is_blocking
                for finding in normalized_findings
            ),
            critical_finding_count=sum(
                finding.severity == SecuritySeverity.CRITICAL
                for finding in normalized_findings
            ),
            high_finding_count=sum(
                finding.severity == SecuritySeverity.HIGH
                for finding in normalized_findings
            ),
            medium_finding_count=sum(
                finding.severity == SecuritySeverity.MEDIUM
                for finding in normalized_findings
            ),
            low_finding_count=sum(
                finding.severity == SecuritySeverity.LOW
                for finding in normalized_findings
            ),
            info_finding_count=sum(
                finding.severity == SecuritySeverity.INFO
                for finding in normalized_findings
            ),
            analyzer_names=normalized_analyzers,
            findings=normalized_findings,
            metadata={
                **(metadata or {}),
                "risk_threshold": self._risk_threshold,
                "risk_weights": {
                    "critical": self._weights.critical,
                    "high": self._weights.high,
                    "medium": self._weights.medium,
                    "low": self._weights.low,
                    "info": self._weights.info,
                },
            },
        )

    def assess_reports(
        self,
        reports: Iterable[SecurityReport],
        *,
        metadata: dict[str, object] | None = None,
    ) -> RiskAssessment:
        """Combine multiple security reports into one assessment."""
        normalized_reports = tuple(reports)

        if any(
            not isinstance(report, SecurityReport)
            for report in normalized_reports
        ):
            raise RiskEngineInputError(
                "All reports must be SecurityReport instances"
            )

        findings: list[SecurityFinding] = []
        analyzer_names: list[str] = []

        for report in normalized_reports:
            findings.extend(report.findings)

            if report.analyzer_name.strip():
                analyzer_names.append(report.analyzer_name)

        return self.assess_findings(
            findings,
            analyzer_names=analyzer_names,
            metadata={
                **(metadata or {}),
                "report_count": len(normalized_reports),
            },
        )

    def assess_report(
        self,
        report: SecurityReport,
    ) -> RiskAssessment:
        """Assess one security report."""
        if not isinstance(report, SecurityReport):
            raise RiskEngineInputError(
                "report must be a SecurityReport"
            )

        return self.assess_reports((report,))

    def _calculate_risk_score(
        self,
        findings: tuple[SecurityFinding, ...],
    ) -> float:
        """Calculate a bounded risk score between 0 and 1."""
        if not findings:
            return 0.0

        weighted_risk = sum(
            self._weights.for_severity(finding.severity)
            * max(0.0, min(1.0, finding.confidence))
            for finding in findings
        )

        normalization = max(
            self._weights.critical,
            1.0,
        )

        return min(
            1.0,
            weighted_risk / (normalization * len(findings)),
        )

    @staticmethod
    def merge_assessments(
        assessments: Iterable[RiskAssessment],
    ) -> RiskAssessment:
        """Merge previously calculated assessments."""
        normalized = tuple(assessments)

        if not normalized:
            raise RiskEngineInputError(
                "At least one assessment is required"
            )

        findings: list[SecurityFinding] = []

        analyzer_names: list[str] = []

        for assessment in normalized:
            findings.extend(assessment.findings)
            analyzer_names.extend(assessment.analyzer_names)

        highest_decision = SecurityDecision.PASS

        for assessment in normalized:
            if assessment.decision == SecurityDecision.REJECT:
                highest_decision = SecurityDecision.REJECT
                break

            if (
                assessment.decision == SecurityDecision.FLAG
                and highest_decision == SecurityDecision.PASS
            ):
                highest_decision = SecurityDecision.FLAG

        risk_score = max(
            assessment.risk_score
            for assessment in normalized
        )

        return RiskAssessment(
            decision=highest_decision,
            risk_score=risk_score,
            finding_count=len(findings),
            blocking_finding_count=sum(
                finding.is_blocking
                for finding in findings
            ),
            critical_finding_count=sum(
                finding.severity == SecuritySeverity.CRITICAL
                for finding in findings
            ),
            high_finding_count=sum(
                finding.severity == SecuritySeverity.HIGH
                for finding in findings
            ),
            medium_finding_count=sum(
                finding.severity == SecuritySeverity.MEDIUM
                for finding in findings
            ),
            low_finding_count=sum(
                finding.severity == SecuritySeverity.LOW
                for finding in findings
            ),
            info_finding_count=sum(
                finding.severity == SecuritySeverity.INFO
                for finding in findings
            ),
            analyzer_names=tuple(
                dict.fromkeys(analyzer_names)
            ),
            findings=tuple(findings),
            metadata={
                "merged_assessment_count": len(normalized),
            },
        )


__all__ = [
    "RiskAssessment",
    "RiskEngineError",
    "RiskEngineInputError",
    "RiskWeights",
    "SecurityRiskEngine",
]