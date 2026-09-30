from __future__ import annotations

import pytest

from src.models import SecurityDecision, SecuritySeverity
from src.security.decision import (
    SecurityDecisionEngine,
    SecurityDecisionError,
    SecurityDecisionPolicy,
)
from src.security.findings import (
    SecurityFinding,
    SecurityLocation,
    SecurityReport,
)


def make_finding(
    *,
    severity: SecuritySeverity = SecuritySeverity.MEDIUM,
    decision: SecurityDecision = SecurityDecision.FLAG,
    finding_id: str = "F001",
) -> SecurityFinding:
    return SecurityFinding(
        finding_id=finding_id,
        category="test",
        rule_id="TEST-001",
        title="Test security finding",
        description="Synthetic finding for unit testing.",
        severity=severity,
        confidence=0.95,
        decision=decision,
        location=SecurityLocation(
            file_name="example.py",
            line_start=1,
            line_end=1,
        ),
        evidence="synthetic test evidence",
        remediation="remove the unsafe construct",
        source="unit-test",
    )


class TestSecurityDecisionPolicy:
    def test_default_policy_contains_reject_severities(self):
        policy = SecurityDecisionPolicy()

        assert SecuritySeverity.CRITICAL in policy.reject_severities
        assert SecuritySeverity.HIGH in policy.reject_severities

    def test_default_policy_contains_flag_severities(self):
        policy = SecurityDecisionPolicy()

        assert SecuritySeverity.MEDIUM in policy.flag_severities
        assert SecuritySeverity.LOW in policy.flag_severities
        assert SecuritySeverity.INFO in policy.flag_severities

    def test_overlapping_severities_are_rejected(self):
        with pytest.raises(SecurityDecisionError):
            SecurityDecisionPolicy(
                reject_severities=frozenset({SecuritySeverity.HIGH}),
                flag_severities=frozenset({SecuritySeverity.HIGH}),
            )


class TestSecurityDecisionEngine:
    def test_critical_finding_is_rejected(self):
        engine = SecurityDecisionEngine()

        finding = make_finding(
            severity=SecuritySeverity.CRITICAL,
            decision=SecurityDecision.FLAG,
        )

        assert engine.decide_finding(finding) == SecurityDecision.REJECT

    def test_high_finding_is_rejected(self):
        engine = SecurityDecisionEngine()

        finding = make_finding(
            severity=SecuritySeverity.HIGH,
            decision=SecurityDecision.FLAG,
        )

        assert engine.decide_finding(finding) == SecurityDecision.REJECT

    def test_medium_finding_is_flagged(self):
        engine = SecurityDecisionEngine()

        finding = make_finding(
            severity=SecuritySeverity.MEDIUM,
            decision=SecurityDecision.FLAG,
        )

        assert engine.decide_finding(finding) == SecurityDecision.FLAG

    def test_low_finding_is_flagged(self):
        engine = SecurityDecisionEngine()

        finding = make_finding(
            severity=SecuritySeverity.LOW,
            decision=SecurityDecision.FLAG,
        )

        assert engine.decide_finding(finding) == SecurityDecision.FLAG

    def test_info_finding_is_flagged(self):
        engine = SecurityDecisionEngine()

        finding = make_finding(
            severity=SecuritySeverity.INFO,
            decision=SecurityDecision.FLAG,
        )

        assert engine.decide_finding(finding) == SecurityDecision.FLAG

    def test_empty_findings_pass(self):
        engine = SecurityDecisionEngine()

        assert engine.decide_findings([]) == SecurityDecision.PASS

    def test_only_safe_findings_pass(self):
        engine = SecurityDecisionEngine()

        finding = make_finding(
            severity=SecuritySeverity.INFO,
            decision=SecurityDecision.PASS,
        )

        assert engine.decide_findings([finding]) == SecurityDecision.FLAG

    def test_reject_has_priority_over_flag(self):
        engine = SecurityDecisionEngine()

        findings = [
            make_finding(
                severity=SecuritySeverity.MEDIUM,
                decision=SecurityDecision.FLAG,
                finding_id="F001",
            ),
            make_finding(
                severity=SecuritySeverity.HIGH,
                decision=SecurityDecision.FLAG,
                finding_id="F002",
            ),
        ]

        assert engine.decide_findings(findings) == SecurityDecision.REJECT

    def test_flag_has_priority_over_pass(self):
        engine = SecurityDecisionEngine()

        findings = [
            make_finding(
                severity=SecuritySeverity.INFO,
                decision=SecurityDecision.PASS,
                finding_id="F001",
            ),
            make_finding(
                severity=SecuritySeverity.MEDIUM,
                decision=SecurityDecision.FLAG,
                finding_id="F002",
            ),
        ]

        assert engine.decide_findings(findings) == SecurityDecision.FLAG

    def test_invalid_finding_is_rejected(self):
        engine = SecurityDecisionEngine()

        with pytest.raises(SecurityDecisionError):
            engine.decide_finding("not-a-finding")  # type: ignore[arg-type]

    def test_none_findings_are_rejected(self):
        engine = SecurityDecisionEngine()

        with pytest.raises(SecurityDecisionError):
            engine.decide_findings(None)  # type: ignore[arg-type]

    def test_invalid_report_is_rejected(self):
        engine = SecurityDecisionEngine()

        with pytest.raises(SecurityDecisionError):
            engine.decide_report(None)  # type: ignore[arg-type]

    def test_report_decision_is_recalculated(self):
        engine = SecurityDecisionEngine()

        finding = make_finding(
            severity=SecuritySeverity.HIGH,
            decision=SecurityDecision.FLAG,
        )

        report = SecurityReport(
            findings=(finding,),
            decision=SecurityDecision.FLAG,
            analyzer_name="test-analyzer",
        )

        assert engine.decide_report(report) == SecurityDecision.REJECT

    def test_evaluate_returns_security_report(self):
        engine = SecurityDecisionEngine()

        finding = make_finding(
            severity=SecuritySeverity.HIGH,
            decision=SecurityDecision.FLAG,
        )

        report = engine.evaluate([finding])

        assert isinstance(report, SecurityReport)
        assert report.decision == SecurityDecision.REJECT
        assert report.finding_count == 1
        assert report.analyzer_name == "security-decision-engine"

    def test_evaluate_preserves_findings(self):
        engine = SecurityDecisionEngine()

        finding = make_finding(
            severity=SecuritySeverity.MEDIUM,
            decision=SecurityDecision.FLAG,
        )

        report = engine.evaluate([finding])

        assert len(report.findings) == 1
        assert report.findings[0] == finding

    def test_evaluate_rejects_invalid_finding_collection(self):
        engine = SecurityDecisionEngine()

        with pytest.raises(SecurityDecisionError):
            engine.evaluate([object()])  # type: ignore[list-item]

    def test_custom_policy_can_reject_medium(self):
        policy = SecurityDecisionPolicy(
            reject_severities=frozenset(
                {
                    SecuritySeverity.CRITICAL,
                    SecuritySeverity.HIGH,
                    SecuritySeverity.MEDIUM,
                }
            ),
            flag_severities=frozenset(
                {
                    SecuritySeverity.LOW,
                    SecuritySeverity.INFO,
                }
            ),
        )

        engine = SecurityDecisionEngine(policy)

        finding = make_finding(
            severity=SecuritySeverity.MEDIUM,
            decision=SecurityDecision.FLAG,
        )

        assert engine.decide_finding(finding) == SecurityDecision.REJECT

    def test_custom_policy_can_allow_medium_as_flag(self):
        policy = SecurityDecisionPolicy(
            reject_severities=frozenset({SecuritySeverity.CRITICAL}),
            flag_severities=frozenset(
                {
                    SecuritySeverity.HIGH,
                    SecuritySeverity.MEDIUM,
                    SecuritySeverity.LOW,
                    SecuritySeverity.INFO,
                }
            ),
        )

        engine = SecurityDecisionEngine(policy)

        finding = make_finding(
            severity=SecuritySeverity.HIGH,
            decision=SecurityDecision.FLAG,
        )

        assert engine.decide_finding(finding) == SecurityDecision.FLAG

    def test_blocking_finding_is_rejected(self):
        engine = SecurityDecisionEngine()

        finding = make_finding(
            severity=SecuritySeverity.LOW,
            decision=SecurityDecision.REJECT,
        )

        assert engine.decide_finding(finding) == SecurityDecision.REJECT

    def test_blocking_policy_can_be_disabled(self):
        policy = SecurityDecisionPolicy(
            reject_severities=frozenset({SecuritySeverity.CRITICAL}),
            flag_severities=frozenset(
                {
                    SecuritySeverity.HIGH,
                    SecuritySeverity.MEDIUM,
                    SecuritySeverity.LOW,
                    SecuritySeverity.INFO,
                }
            ),
            reject_on_blocking_finding=False,
        )

        engine = SecurityDecisionEngine(policy)

        finding = make_finding(
            severity=SecuritySeverity.LOW,
            decision=SecurityDecision.REJECT,
        )

        assert engine.decide_finding(finding) == SecurityDecision.FLAG

    def test_policy_property_returns_active_policy(self):
        policy = SecurityDecisionPolicy()

        engine = SecurityDecisionEngine(policy)

        assert engine.policy is policy