from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models import SecurityDecision, SecuritySeverity
from src.security.findings import (
    SecurityFinding,
    SecurityLocation,
    SecurityReport,
)


class SemgrepRunnerError(RuntimeError):
    """Base exception for Semgrep runner failures."""


class SemgrepConfigurationError(SemgrepRunnerError):
    """Raised when the Semgrep runner configuration is invalid."""


class SemgrepExecutionError(SemgrepRunnerError):
    """Raised when Semgrep execution fails unexpectedly."""


@dataclass(frozen=True)
class SemgrepConfig:
    """Configuration for invoking the Semgrep CLI."""

    executable: str = "semgrep"
    config: str = "auto"
    timeout_seconds: int = 60
    max_target_bytes: int = 2_000_000
    additional_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.executable.strip():
            raise SemgrepConfigurationError(
                "Semgrep executable cannot be empty"
            )

        if not self.config.strip():
            raise SemgrepConfigurationError(
                "Semgrep config cannot be empty"
            )

        if self.timeout_seconds <= 0:
            raise SemgrepConfigurationError(
                "timeout_seconds must be greater than zero"
            )

        if self.max_target_bytes <= 0:
            raise SemgrepConfigurationError(
                "max_target_bytes must be greater than zero"
            )

        for argument in self.additional_args:
            if not isinstance(argument, str) or not argument:
                raise SemgrepConfigurationError(
                    "additional_args must contain non-empty strings"
                )


@dataclass(frozen=True)
class SemgrepResult:
    """Raw result information returned by the Semgrep runner."""

    available: bool
    findings: tuple[SecurityFinding, ...] = ()
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float | None = None
    error: str | None = None


class SemgrepRunner:
    """Execute Semgrep and normalize its findings."""

    def __init__(
        self,
        config: SemgrepConfig | None = None,
    ) -> None:
        self._config = config or SemgrepConfig()

    @property
    def config(self) -> SemgrepConfig:
        """Return the active Semgrep configuration."""
        return self._config

    def is_available(self) -> bool:
        """Return whether the configured Semgrep executable is available."""
        return shutil.which(self._config.executable) is not None

    def version(self) -> str | None:
        """Return the installed Semgrep version when available."""
        if not self.is_available():
            return None

        try:
            completed = subprocess.run(
                [self._config.executable, "--version"],
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        if completed.returncode != 0:
            return None

        version = completed.stdout.strip()
        return version or None

    def analyze_code(
        self,
        code: str,
        *,
        file_name: str = "generated_code.py",
        language: str | None = None,
    ) -> SemgrepResult:
        """Analyze an in-memory source string with Semgrep."""
        if not isinstance(code, str):
            raise SemgrepRunnerError("code must be a string")

        if not code.strip():
            raise SemgrepRunnerError("code cannot be empty")

        if not file_name.strip():
            raise SemgrepRunnerError("file_name cannot be empty")

        encoded_size = len(code.encode("utf-8"))

        if encoded_size > self._config.max_target_bytes:
            raise SemgrepRunnerError(
                "code exceeds the configured Semgrep target size"
            )

        if not self.is_available():
            return SemgrepResult(
                available=False,
                error="Semgrep executable was not found",
            )

        suffix = Path(file_name).suffix or ".txt"

        with tempfile.TemporaryDirectory(
            prefix="securecoderag-semgrep-"
        ) as temporary_directory:
            target = Path(temporary_directory) / f"target{suffix}"
            target.write_text(code, encoding="utf-8")

            return self._run_target(
                target,
                language=language,
            )

    def analyze_file(
        self,
        file_path: str | Path,
        *,
        language: str | None = None,
    ) -> SemgrepResult:
        """Analyze an existing source file with Semgrep."""
        target = Path(file_path)

        if not target.exists():
            raise SemgrepRunnerError(
                f"Target file does not exist: {target}"
            )

        if not target.is_file():
            raise SemgrepRunnerError(
                f"Target path is not a file: {target}"
            )

        if target.stat().st_size > self._config.max_target_bytes:
            raise SemgrepRunnerError(
                "target file exceeds the configured Semgrep target size"
            )

        if not self.is_available():
            return SemgrepResult(
                available=False,
                error="Semgrep executable was not found",
            )

        return self._run_target(
            target,
            language=language,
        )

    def _run_target(
        self,
        target: Path,
        *,
        language: str | None,
    ) -> SemgrepResult:
        command = self._build_command(
            target,
            language=language,
        )

        from time import perf_counter

        started = perf_counter()

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = (perf_counter() - started) * 1000

            return SemgrepResult(
                available=True,
                return_code=None,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                duration_ms=duration_ms,
                error=(
                    "Semgrep execution timed out after "
                    f"{self._config.timeout_seconds} seconds"
                ),
            )
        except OSError as exc:
            raise SemgrepExecutionError(
                f"Failed to execute Semgrep: {exc}"
            ) from exc

        duration_ms = (perf_counter() - started) * 1000

        if not completed.stdout.strip():
            if completed.returncode == 0:
                findings: tuple[SecurityFinding, ...] = ()
                error = None
            else:
                findings = ()
                error = (
                    completed.stderr.strip()
                    or f"Semgrep exited with code {completed.returncode}"
                )

            return SemgrepResult(
                available=True,
                findings=findings,
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_ms=duration_ms,
                error=error,
            )

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise SemgrepExecutionError(
                "Semgrep returned invalid JSON output"
            ) from exc

        findings = self._parse_findings(payload)

        error = None

        if completed.returncode not in (0, 1):
            error = (
                completed.stderr.strip()
                or f"Semgrep exited with code {completed.returncode}"
            )

        return SemgrepResult(
            available=True,
            findings=findings,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=duration_ms,
            error=error,
        )

    def _build_command(
        self,
        target: Path,
        *,
        language: str | None,
    ) -> list[str]:
        command = [
            self._config.executable,
            "--json",
            "--config",
            self._config.config,
        ]

        if language:
            command.extend(["--lang", language])

        command.extend(self._config.additional_args)
        command.append(str(target))

        return command

    def _parse_findings(
        self,
        payload: dict[str, Any],
    ) -> tuple[SecurityFinding, ...]:
        results = payload.get("results", [])

        if not isinstance(results, list):
            raise SemgrepExecutionError(
                "Semgrep JSON output contains an invalid results field"
            )

        findings: list[SecurityFinding] = []

        for index, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue

            findings.append(
                self._convert_result(
                    result,
                    index=index,
                )
            )

        return tuple(findings)

    def _convert_result(
        self,
        result: dict[str, Any],
        *,
        index: int,
    ) -> SecurityFinding:
        check = result.get("check_id", f"SEMGREP-{index:04d}")

        extra = result.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}

        metadata = extra.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        severity = self._map_severity(
            extra.get("severity") or metadata.get("severity")
        )

        confidence = self._map_confidence(
            metadata.get("confidence")
        )

        path = str(result.get("path", "<unknown>"))

        start = result.get("start", {})
        end = result.get("end", {})

        if not isinstance(start, dict):
            start = {}

        if not isinstance(end, dict):
            end = {}

        line_start = self._optional_int(start.get("line"))
        line_end = self._optional_int(end.get("line"))
        column_start = self._optional_int(start.get("col"))
        column_end = self._optional_int(end.get("col"))

        message = str(
            extra.get("message")
            or result.get("message")
            or "Semgrep security finding"
        )

        lines = extra.get("lines")
        if lines is None:
            lines = result.get("match")

        evidence = str(lines) if lines is not None else message

        decision = (
            SecurityDecision.REJECT
            if severity
            in {
                SecuritySeverity.CRITICAL,
                SecuritySeverity.HIGH,
            }
            else SecurityDecision.FLAG
        )

        return SecurityFinding(
            finding_id=f"semgrep-{index:04d}",
            category="static-analysis",
            rule_id=str(check),
            title=message,
            description=message,
            severity=severity,
            confidence=confidence,
            decision=decision,
            location=SecurityLocation(
                file_name=path,
                line_start=line_start,
                line_end=line_end,
                column_start=column_start,
                column_end=column_end,
            ),
            evidence=evidence,
            remediation=self._extract_remediation(extra, metadata),
            source="semgrep",
            metadata={
                "semgrep_check_id": str(check),
                "metadata": metadata,
                "raw_extra": extra,
            },
        )

    @staticmethod
    def _map_severity(value: Any) -> SecuritySeverity:
        normalized = str(value or "").strip().lower()

        mapping = {
            "critical": SecuritySeverity.CRITICAL,
            "error": SecuritySeverity.HIGH,
            "high": SecuritySeverity.HIGH,
            "warning": SecuritySeverity.MEDIUM,
            "medium": SecuritySeverity.MEDIUM,
            "low": SecuritySeverity.LOW,
            "info": SecuritySeverity.INFO,
            "information": SecuritySeverity.INFO,
        }

        return mapping.get(
            normalized,
            SecuritySeverity.MEDIUM,
        )

    @staticmethod
    def _map_confidence(value: Any) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))

        normalized = str(value or "").strip().lower()

        mapping = {
            "very high": 0.95,
            "high": 0.85,
            "medium": 0.70,
            "low": 0.50,
            "very low": 0.25,
        }

        return mapping.get(normalized, 0.70)

    @staticmethod
    def _extract_remediation(
        extra: dict[str, Any],
        metadata: dict[str, Any],
    ) -> str:
        fix = extra.get("fix")

        if fix:
            return str(fix)

        fix_regex = extra.get("fix-regex")
        if fix_regex:
            return str(fix_regex)

        remediation = metadata.get("fix")
        if remediation:
            return str(remediation)

        return "Review the Semgrep finding and apply the recommended secure fix."

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def analyze_to_report(
        self,
        code: str,
        *,
        file_name: str = "generated_code.py",
        language: str | None = None,
    ) -> SecurityReport:
        """Analyze source code and return a normalized security report."""
        result = self.analyze_code(
            code,
            file_name=file_name,
            language=language,
        )

        if not result.available:
            return SecurityReport(
                findings=(),
                decision=SecurityDecision.PASS,
                analyzer_name="semgrep",
                metadata={
                    "available": False,
                    "error": result.error,
                    "duration_ms": result.duration_ms,
                },
            )

        decision = self._aggregate_decision(result.findings)

        return SecurityReport(
            findings=result.findings,
            decision=decision,
            analyzer_name="semgrep",
            metadata={
                "available": True,
                "return_code": result.return_code,
                "duration_ms": result.duration_ms,
                "error": result.error,
                "version": self.version(),
            },
        )

    @staticmethod
    def _aggregate_decision(
        findings: Sequence[SecurityFinding],
    ) -> SecurityDecision:
        if any(
            finding.decision == SecurityDecision.REJECT
            for finding in findings
        ):
            return SecurityDecision.REJECT

        if findings:
            return SecurityDecision.FLAG

        return SecurityDecision.PASS


__all__ = [
    "SemgrepConfig",
    "SemgrepConfigurationError",
    "SemgrepExecutionError",
    "SemgrepResult",
    "SemgrepRunner",
    "SemgrepRunnerError",
]