"""Dataset models and builders for SecureCodeRAG poisoning experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


class BenchmarkDatasetError(Exception):
    """Base exception for benchmark dataset failures."""


class BenchmarkDatasetInputError(BenchmarkDatasetError):
    """Raised when benchmark dataset input is invalid."""


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """A single clean/poisoned benchmark sample."""

    sample_id: str
    query: str
    clean_code: str
    poisoned_code: str
    language: str
    poisoning_category: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = {
            "sample_id": self.sample_id,
            "query": self.query,
            "clean_code": self.clean_code,
            "poisoned_code": self.poisoned_code,
            "language": self.language,
            "poisoning_category": self.poisoning_category,
        }

        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise BenchmarkDatasetInputError(
                    f"{name} must be a non-empty string."
                )

        if not isinstance(self.metadata, dict):
            raise BenchmarkDatasetInputError(
                "metadata must be a dictionary."
            )

    @property
    def is_modified(self) -> bool:
        """Return whether the poisoned sample differs from the clean sample."""
        return self.clean_code != self.poisoned_code

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "clean_code": self.clean_code,
            "poisoned_code": self.poisoned_code,
            "language": self.language,
            "poisoning_category": self.poisoning_category,
            "metadata": dict(self.metadata),
        }


class BenchmarkDataset:
    """In-memory collection of benchmark samples."""

    def __init__(
        self,
        samples: Iterable[BenchmarkSample] | None = None,
    ) -> None:
        self._samples: list[BenchmarkSample] = []

        if samples is not None:
            for sample in samples:
                self.add(sample)

    @property
    def size(self) -> int:
        """Return the number of samples."""
        return len(self._samples)

    @property
    def samples(self) -> tuple[BenchmarkSample, ...]:
        """Return immutable sample view."""
        return tuple(self._samples)

    def add(self, sample: BenchmarkSample) -> None:
        """Add one benchmark sample."""
        if not isinstance(sample, BenchmarkSample):
            raise BenchmarkDatasetInputError(
                "sample must be a BenchmarkSample."
            )

        if any(existing.sample_id == sample.sample_id for existing in self._samples):
            raise BenchmarkDatasetInputError(
                f"Duplicate sample ID: {sample.sample_id}"
            )

        self._samples.append(sample)

    def get(self, sample_id: str) -> BenchmarkSample:
        """Retrieve a sample by ID."""
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise BenchmarkDatasetInputError(
                "sample_id must be a non-empty string."
            )

        for sample in self._samples:
            if sample.sample_id == sample_id:
                return sample

        raise BenchmarkDatasetError(
            f"Benchmark sample not found: {sample_id}"
        )

    def by_category(
        self,
        category: str,
    ) -> list[BenchmarkSample]:
        """Return samples belonging to one poisoning category."""
        return [
            sample
            for sample in self._samples
            if sample.poisoning_category == category
        ]

    def categories(self) -> list[str]:
        """Return unique poisoning categories."""
        return sorted(
            {
                sample.poisoning_category
                for sample in self._samples
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataset."""
        return {
            "size": self.size,
            "categories": self.categories(),
            "samples": [
                sample.to_dict()
                for sample in self._samples
            ],
        }

    def __len__(self) -> int:
        return self.size