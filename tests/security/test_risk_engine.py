from __future__ import annotations

import pytest

from src.models import SecurityDecision, SecuritySeverity
from src.security.decision import SecurityDecisionEngine
from src.security.findings import (
    SecurityFinding,
    SecurityLocation,
    SecurityReport,
)
from src.security.risk_engine import (
    RiskEngineInputError,
    RiskWeights,
    SecurityRiskEngine,
)


def make_finding(
    *,
    finding_id: str = "finding-1",
    severity: SecuritySeverity = SecuritySeverity.MEDIUM,
    confidence: float = 1.0,
    decision: SecurityDecision = SecurityDecision.FLAG,
    analyzer: str = "test",
) -> SecurityFinding:
    return SecurityFinding(
        finding_id=finding_id,
        category="test",
        rule_id=f"rule-{finding_id}",
        title="Test security finding",
        description="Test security finding",
        severity=severity,
        confidence=confidence,
        decision=decision,
        location=SecurityLocation(
            file_name="example.py",
            line_start=1,
            line_end=1,
        ),
        evidence="test evidence",
        remediation="test remediation",
        source=analyzer,
    )


def make_report(
    *,
    findings: tuple[SecurityFinding, ...] = (),
    analyzer_name: str = "test-analyzer",
    decision: SecurityDecision = SecurityDecision.PASS,
) -> SecurityReport:
    return SecurityReport(
        findings=findings,
        decision=decision,
        analyzer_name=analyzer_name,
    )


class TestRiskWeights:
    def test_default_weights(self):
        weights = RiskWeights()

        assert weights.critical == 1.00
        assert weights.high == 0.80
        assert weights.medium == 0.50
        assert weights.low == 0.20
        assert weights.info == 0.05

    def test_maps_each_severity(self):
        weights = RiskWeights()

        assert (
            weights.for_severity(SecuritySeverity.CRITICAL)
            == 1.00
        )
        assert (
            weights.for_severity(SecuritySeverity.HIGH)
            == 0.80
        )
        assert (
            weights.for_severity(SecuritySeverity.MEDIUM)
            == 0.50
        )
        assert (
            weights.for_severity(SecuritySeverity.LOW)
            == 0.20
        )
        assert (
            weights.for_severity(SecuritySeverity.INFO)
            == 0.05
        )

    def test_rejects_negative_weight(self):
        with pytest.raises(RiskEngineInputError):
            RiskWeights(critical=-1.0)

    def test_rejects_incorrect_critical_order(self):
        with pytest.raises(RiskEngineInputError):
            RiskWeights(
                critical=0.5,
                high=0.8,
            )

    def test_rejects_incorrect_high_order(self):
        with pytest.raises(RiskEngineInputError):
            RiskWeights(
                high=0.4,
                medium=0.5,
            )

    def test_rejects_incorrect_medium_order(self):
        with pytest.raises(RiskEngineInputError):
            RiskWeights(
                medium=0.1,
                low=0.2,
            )

    def test_rejects_incorrect_low_order(self):
        with pytest.raises(RiskEngineInputError):
            RiskWeights(
                low=0.01,
                info=0.05,
            )


class TestSecurityRiskEngine:
    def test_default_configuration(self):
        engine = SecurityRiskEngine()

        assert engine.risk_threshold == 0.50
        assert engine.weights.critical == 1.00
        assert isinstance(
            engine.decision_engine,
            SecurityDecisionEngine,
        )

    def test_rejects_invalid_threshold(self):
        with pytest.raises(RiskEngineInputError):
            SecurityRiskEngine(risk_threshold=-0.1)

        with pytest.raises(RiskEngineInputError):
            SecurityRiskEngine(risk_threshold=1.1)

    def test_empty_findings_pass(self):
        engine = SecurityRiskEngine()

        assessment = engine.assess_findings(())

        assert assessment.decision == SecurityDecision.PASS
        assert assessment.risk_score == 0.0
        assert assessment.finding_count == 0
        assert assessment.blocking_finding_count == 0
        assert assessment.is_safe is True
        assert assessment.requires_review is False
        assert assessment.must_be_rejected is False

    def test_high_finding_is_rejected(self):
        engine = SecurityRiskEngine()

        finding = make_finding(
            severity=SecuritySeverity.HIGH,
            decision=SecurityDecision.REJECT,
        )

        assessment = engine.assess_findings(
            [finding],
            analyzer_names=["ast"],
        )

        assert assessment.decision == SecurityDecision.REJECT
        assert assessment.finding_count == 1
        assert assessment.high_finding_count == 1
        assert assessment.must_be_rejected is True
        assert assessment.is_safe is False

    def test_medium_finding_is_flagged(self):
        engine = SecurityRiskEngine()

        finding = make_finding(
            severity=SecuritySeverity.MEDIUM,
            decision=SecurityDecision.FLAG,
        )

        assessment = engine.assess_findings(
            [finding],
            analyzer_names=["semgrep"],
        )

        assert assessment.decision == SecurityDecision.FLAG
        assert assessment.risk_score == 0.50
        assert assessment.medium_finding_count == 1
        assert assessment.requires_review is True

    def test_low_finding_below_threshold_remains_flagged_by_policy(self):
        engine = SecurityRiskEngine()

        finding = make_finding(
            severity=SecuritySeverity.LOW,
            decision=SecurityDecision.FLAG,
        )

        assessment = engine.assess_findings([finding])

        assert assessment.decision == SecurityDecision.FLAG
        assert assessment.risk_score == 0.20
        assert assessment.low_finding_count == 1

    def test_info_finding_is_flagged_by_default_policy(self):
        engine = SecurityRiskEngine()

        finding = make_finding(
            severity=SecuritySeverity.INFO,
            decision=SecurityDecision.FLAG,
        )

        assessment = engine.assess_findings([finding])

        assert assessment.decision == SecurityDecision.FLAG
        assert assessment.risk_score == 0.05
        assert assessment.info_finding_count == 1

    def test_risk_score_uses_confidence(self):
        engine = SecurityRiskEngine()

        finding = make_finding(
            severity=SecuritySeverity.HIGH,
            confidence=0.50,
            decision=SecurityDecision.FLAG,
        )

        assessment = engine.assess_findings([finding])

        assert assessment.risk_score == pytest.approx(0.40)
        assert assessment.decision == SecurityDecision.REJECT

    def test_risk_score_is_bounded_to_one(self):
        engine = SecurityRiskEngine()

        findings = [
            make_finding(
                finding_id="one",
                severity=SecuritySeverity.CRITICAL,
                confidence=1.0,
                decision=SecurityDecision.REJECT,
            ),
            make_finding(
                finding_id="two",
                severity=SecuritySeverity.CRITICAL,
                confidence=1.0,
                decision=SecurityDecision.REJECT,
            ),
        ]

        assessment = engine.assess_findings(findings)

        assert 0.0 <= assessment.risk_score <= 1.0
        assert assessment.risk_score == 1.0

    def test_counts_all_severities(self):
        engine = SecurityRiskEngine()

        findings = [
            make_finding(
                finding_id="critical",
                severity=SecuritySeverity.CRITICAL,
                decision=SecurityDecision.REJECT,
            ),
            make_finding(
                finding_id="high",
                severity=SecuritySeverity.HIGH,
                decision=SecurityDecision.REJECT,
            ),
            make_finding(
                finding_id="medium",
                severity=SecuritySeverity.MEDIUM,
                decision=SecurityDecision.FLAG,
            ),
            make_finding(
                finding_id="low",
                severity=SecuritySeverity.LOW,
                decision=SecurityDecision.FLAG,
            ),
            make_finding(
                finding_id="info",
                severity=SecuritySeverity.INFO,
                decision=SecurityDecision.FLAG,
            ),
        ]

        assessment = engine.assess_findings(findings)

        assert assessment.finding_count == 5
        assert assessment.critical_finding_count == 1
        assert assessment.high_finding_count == 1
        assert assessment.medium_finding_count == 1
        assert assessment.low_finding_count == 1
        assert assessment.info_finding_count == 1
        assert assessment.decision == SecurityDecision.REJECT

    def test_preserves_analyzer_names_without_duplicates(self):
        engine = SecurityRiskEngine()

        finding = make_finding()

        assessment = engine.assess_findings(
            [finding],
            analyzer_names=[
                "ast",
                "semgrep",
                "ast",
                "",
                "  ",
            ],
        )

        assert assessment.analyzer_names == (
            "ast",
            "semgrep",
        )

    def test_preserves_metadata(self):
        engine = SecurityRiskEngine()

        assessment = engine.assess_findings(
            [],
            metadata={
                "request_id": "request-123",
                "defense_layer": "L5",
            },
        )

        assert assessment.metadata["request_id"] == "request-123"
        assert assessment.metadata["defense_layer"] == "L5"
        assert assessment.metadata["risk_threshold"] == 0.50

    def test_rejects_invalid_finding_type(self):
        engine = SecurityRiskEngine()

        with pytest.raises(RiskEngineInputError):
            engine.assess_findings(
                [object()],
            )

    def test_assess_report(self):
        engine = SecurityRiskEngine()

        finding = make_finding(
            severity=SecuritySeverity.HIGH,
            decision=SecurityDecision.REJECT,
        )

        report = make_report(
            findings=(finding,),
            analyzer_name="ast",
            decision=SecurityDecision.REJECT,
        )

        assessment = engine.assess_report(report)

        assert assessment.decision == SecurityDecision.REJECT
        assert assessment.finding_count == 1
        assert assessment.analyzer_names == ("ast",)

    def test_rejects_invalid_report(self):
        engine = SecurityRiskEngine()

        with pytest.raises(RiskEngineInputError):
            engine.assess_report(object())

    def test_assess_reports_combines_findings(self):
        engine = SecurityRiskEngine()

        ast_finding = make_finding(
            finding_id="ast-finding",
            severity=SecuritySeverity.HIGH,
            decision=SecurityDecision.REJECT,
            analyzer="ast",
        )

        semgrep_finding = make_finding(
            finding_id="semgrep-finding",
            severity=SecuritySeverity.MEDIUM,
            decision=SecurityDecision.FLAG,
            analyzer="semgrep",
        )

        ast_report = make_report(
            findings=(ast_finding,),
            analyzer_name="ast",
            decision=SecurityDecision.REJECT,
        )

        semgrep_report = make_report(
            findings=(semgrep_finding,),
            analyzer_name="semgrep",
            decision=SecurityDecision.FLAG,
        )

        assessment = engine.assess_reports(
            [ast_report, semgrep_report],
        )

        assert assessment.finding_count == 2
        assert assessment.analyzer_names == (
            "ast",
            "semgrep",
        )
        assert assessment.decision == SecurityDecision.REJECT
        assert assessment.metadata["report_count"] == 2

    def test_rejects_invalid_report_collection(self):
        engine = SecurityRiskEngine()

        with pytest.raises(RiskEngineInputError):
            engine.assess_reports([object()])

    def test_threshold_can_upgrade_pass_to_flag(self):
        engine = SecurityRiskEngine(
            risk_threshold=0.10,
        )

        finding = make_finding(
            severity=SecuritySeverity.LOW,
            confidence=0.50,
            decision=SecurityDecision.PASS,
        )

        assessment = engine.assess_findings([finding])

        assert assessment.risk_score == pytest.approx(0.10)
        assert assessment.decision == SecurityDecision.FLAG

    def test_policy_decision_has_priority_over_threshold(self):
        engine = SecurityRiskEngine(
            risk_threshold=0.99,
        )

        finding = make_finding(
            severity=SecuritySeverity.HIGH,
            confidence=0.10,
            decision=SecurityDecision.REJECT,
        )

        assessment = engine.assess_findings([finding])

        assert assessment.risk_score == pytest.approx(0.08)
        assert assessment.decision == SecurityDecision.REJECT

    def test_merge_assessments(self):
        engine = SecurityRiskEngine()

        first = engine.assess_findings(
            [
                make_finding(
                    finding_id="first",
                    severity=SecuritySeverity.MEDIUM,
                    decision=SecurityDecision.FLAG,
                )
            ],
            analyzer_names=["ast"],
        )

        second = engine.assess_findings(
            [
                make_finding(
                    finding_id="second",
                    severity=SecuritySeverity.HIGH,
                    decision=SecurityDecision.REJECT,
                )
            ],
            analyzer_names=["semgrep"],
        )

        merged = SecurityRiskEngine.merge_assessments(
            [first, second]
        )

        assert merged.decision == SecurityDecision.REJECT
        assert merged.risk_score == max(
            first.risk_score,
            second.risk_score,
        )
        assert merged.finding_count == 2
        assert merged.analyzer_names == (
            "ast",
            "semgrep",
        )

    def test_merge_flag_assessments(self):
        engine = SecurityRiskEngine()

        first = engine.assess_findings(
            [
                make_finding(
                    finding_id="first",
                    severity=SecuritySeverity.MEDIUM,
                    decision=SecurityDecision.FLAG,
                )
            ]
        )

        second = engine.assess_findings(
            [
                make_finding(
                    finding_id="second",
                    severity=SecuritySeverity.LOW,
                    decision=SecurityDecision.FLAG,
                )
            ]
        )

        merged = SecurityRiskEngine.merge_assessments(
            [first, second]
        )

        assert merged.decision == SecurityDecision.FLAG
        assert merged.finding_count == 2

    def test_merge_rejects_empty_collection(self):
        with pytest.raises(RiskEngineInputError):
            SecurityRiskEngine.merge_assessments([])

    def test_merge_preserves_all_findings(self):
        engine = SecurityRiskEngine()

        finding_one = make_finding(
            finding_id="one",
            severity=SecuritySeverity.LOW,
        )
        finding_two = make_finding(
            finding_id="two",
            severity=SecuritySeverity.MEDIUM,
        )

        first = engine.assess_findings([finding_one])
        second = engine.assess_findings([finding_two])

        merged = SecurityRiskEngine.merge_assessments(
            [first, second]
        )

        assert {
            finding.finding_id
            for finding in merged.findings
        } == {"one", "two"}