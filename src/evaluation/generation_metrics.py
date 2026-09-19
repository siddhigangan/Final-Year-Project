from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


class GenerationMetricsError(Exception):
    """Base exception for generation metric errors."""


class GenerationMetricsInputError(GenerationMetricsError, ValueError):
    """Raised when generation metric inputs are invalid."""


@dataclass(frozen=True)
class GenerationMetrics:
    """Container for generated-code quality metrics."""

    exact_match: float
    normalized_exact_match: float
    token_precision: float
    token_recall: float
    token_f1: float
    character_similarity: float
    output_nonempty: bool
    utility: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return metrics as a serializable dictionary."""
        return {
            "exact_match": self.exact_match,
            "normalized_exact_match": self.normalized_exact_match,
            "token_precision": self.token_precision,
            "token_recall": self.token_recall,
            "token_f1": self.token_f1,
            "character_similarity": self.character_similarity,
            "output_nonempty": self.output_nonempty,
            "utility": self.utility,
            "metadata": dict(self.metadata),
        }


def _validate_code(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise GenerationMetricsInputError(
            f"{name} must be a string."
        )

    return value


def normalize_code(code: str) -> str:
    """
    Normalize source code for comparison.

    The normalization intentionally does not attempt AST equivalence.
    It removes blank lines, trailing whitespace and common indentation.
    """
    code = _validate_code(code, "code")

    if not code.strip():
        return ""

    lines = [
        line.rstrip()
        for line in code.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    indentation = min(
        (
            len(line) - len(line.lstrip())
            for line in lines
            if line.strip()
        ),
        default=0,
    )

    return "\n".join(
        line[indentation:]
        for line in lines
    ).strip()


def exact_match(
    candidate_code: str,
    reference_code: str,
) -> float:
    """Return 1.0 when candidate and reference match exactly."""
    candidate = _validate_code(candidate_code, "candidate_code")
    reference = _validate_code(reference_code, "reference_code")

    return float(candidate == reference)


def normalized_exact_match(
    candidate_code: str,
    reference_code: str,
) -> float:
    """Return exact-match score after code normalization."""
    return float(
        normalize_code(candidate_code)
        == normalize_code(reference_code)
    )


def _tokenize(code: str) -> list[str]:
    code = _validate_code(code, "code")

    return re.findall(
        r"[A-Za-z_]\w*|\d+(?:\.\d+)?|==|!=|<=|>=|->|=>|"
        r"[^\sA-Za-z0-9_]",
        code,
    )


def token_precision(
    candidate_code: str,
    reference_code: str,
) -> float:
    """Calculate token-level precision."""
    candidate_tokens = _tokenize(candidate_code)
    reference_tokens = _tokenize(reference_code)

    if not candidate_tokens:
        return 0.0

    reference_counts: dict[str, int] = {}

    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1

    matched = 0

    for token in candidate_tokens:
        count = reference_counts.get(token, 0)

        if count > 0:
            matched += 1
            reference_counts[token] = count - 1

    return matched / len(candidate_tokens)


def token_recall(
    candidate_code: str,
    reference_code: str,
) -> float:
    """Calculate token-level recall."""
    candidate_tokens = _tokenize(candidate_code)
    reference_tokens = _tokenize(reference_code)

    if not reference_tokens:
        return 0.0

    candidate_counts: dict[str, int] = {}

    for token in candidate_tokens:
        candidate_counts[token] = candidate_counts.get(token, 0) + 1

    matched = 0

    for token in reference_tokens:
        count = candidate_counts.get(token, 0)

        if count > 0:
            matched += 1
            candidate_counts[token] = count - 1

    return matched / len(reference_tokens)


def token_f1(
    candidate_code: str,
    reference_code: str,
) -> float:
    """Calculate token-level F1."""
    precision = token_precision(candidate_code, reference_code)
    recall = token_recall(candidate_code, reference_code)

    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def character_similarity(
    candidate_code: str,
    reference_code: str,
) -> float:
    """Calculate normalized character-level similarity."""
    candidate = _validate_code(candidate_code, "candidate_code")
    reference = _validate_code(reference_code, "reference_code")

    if candidate == reference:
        return 1.0

    if not candidate and not reference:
        return 1.0

    return SequenceMatcher(
        None,
        candidate,
        reference,
    ).ratio()


def output_is_nonempty(code: str) -> bool:
    """Return whether generated output contains meaningful text."""
    return bool(_validate_code(code, "code").strip())


def calculate_generation_utility(
    candidate_code: str,
    reference_code: str,
) -> float:
    """
    Calculate a deterministic generation utility score.

    Utility combines normalized exact match, token F1 and
    character similarity.
    """
    exact = normalized_exact_match(
        candidate_code,
        reference_code,
    )

    f1 = token_f1(
        candidate_code,
        reference_code,
    )

    similarity = character_similarity(
        candidate_code,
        reference_code,
    )

    return (
        0.50 * exact
        + 0.30 * f1
        + 0.20 * similarity
    )


def calculate_generation_metrics(
    candidate_code: str,
    reference_code: str,
    metadata: dict[str, Any] | None = None,
) -> GenerationMetrics:
    """Calculate the complete generation metric set."""
    return GenerationMetrics(
        exact_match=exact_match(
            candidate_code,
            reference_code,
        ),
        normalized_exact_match=normalized_exact_match(
            candidate_code,
            reference_code,
        ),
        token_precision=token_precision(
            candidate_code,
            reference_code,
        ),
        token_recall=token_recall(
            candidate_code,
            reference_code,
        ),
        token_f1=token_f1(
            candidate_code,
            reference_code,
        ),
        character_similarity=character_similarity(
            candidate_code,
            reference_code,
        ),
        output_nonempty=output_is_nonempty(candidate_code),
        utility=calculate_generation_utility(
            candidate_code,
            reference_code,
        ),
        metadata=dict(metadata or {}),
    )


def batch_generation_utility(
    candidates: Iterable[str],
    references: Iterable[str],
) -> float:
    """
    Calculate mean generation utility across a batch.
    """
    candidate_list = list(candidates)
    reference_list = list(references)

    if len(candidate_list) != len(reference_list):
        raise GenerationMetricsInputError(
            "candidates and references must have the same length."
        )

    if not candidate_list:
        return 0.0

    scores = [
        calculate_generation_utility(
            candidate,
            reference,
        )
        for candidate, reference in zip(
            candidate_list,
            reference_list,
            strict=True,
        )
    ]

    return sum(scores) / len(scores)