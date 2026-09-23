"""Persistent metadata storage for SecureCodeRAG vector indexes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.models import CodeChunk, ProgrammingLanguage, SourceTrust
from src.vectorstore.base import (
    VectorStoreInputError,
    VectorStoreNotFoundError,
)


class MetadataStoreError(Exception):
    """Base exception for metadata-store failures."""


class MetadataStoreCorruptionError(MetadataStoreError):
    """Raised when persisted metadata cannot be loaded safely."""


class MetadataStore:
    """Maintain a persistent mapping between vector IDs and code chunks."""

    FORMAT_VERSION = 1

    def __init__(self) -> None:
        self._records: dict[int, CodeChunk] = {}
        self._chunk_to_vector: dict[str, int] = {}

    @property
    def size(self) -> int:
        """Return the number of stored metadata records."""
        return len(self._records)

    def add(
        self,
        vector_ids: list[int],
        chunks: list[CodeChunk],
    ) -> None:
        """Add vector-to-chunk mappings."""
        if not vector_ids:
            raise VectorStoreInputError(
                "vector_ids cannot be empty."
            )

        if not chunks:
            raise VectorStoreInputError(
                "chunks cannot be empty."
            )

        if len(vector_ids) != len(chunks):
            raise VectorStoreInputError(
                "vector_ids and chunks must have the same length."
            )

        for vector_id, chunk in zip(vector_ids, chunks, strict=True):
            self._validate_vector_id(vector_id)
            self._validate_chunk(chunk)

            if vector_id in self._records:
                raise VectorStoreInputError(
                    f"Vector ID already exists: {vector_id}"
                )

            if chunk.chunk_id in self._chunk_to_vector:
                raise VectorStoreInputError(
                    f"Chunk ID already exists: {chunk.chunk_id}"
                )

            self._records[vector_id] = chunk
            self._chunk_to_vector[chunk.chunk_id] = vector_id

    def get(self, vector_id: int) -> CodeChunk:
        """Return the chunk associated with a vector ID."""
        self._validate_vector_id(vector_id)

        try:
            return self._records[vector_id]
        except KeyError as exc:
            raise VectorStoreNotFoundError(
                f"No metadata found for vector ID: {vector_id}"
            ) from exc

    def get_by_chunk_id(self, chunk_id: str) -> CodeChunk:
        """Return a chunk using its stable chunk ID."""
        self._validate_chunk_id(chunk_id)

        try:
            vector_id = self._chunk_to_vector[chunk_id]
        except KeyError as exc:
            raise VectorStoreNotFoundError(
                f"No metadata found for chunk ID: {chunk_id}"
            ) from exc

        return self.get(vector_id)

    def get_vector_id(self, chunk_id: str) -> int:
        """Return the vector ID associated with a chunk ID."""
        self._validate_chunk_id(chunk_id)

        try:
            return self._chunk_to_vector[chunk_id]
        except KeyError as exc:
            raise VectorStoreNotFoundError(
                f"No vector ID found for chunk ID: {chunk_id}"
            ) from exc

    def contains_vector_id(self, vector_id: int) -> bool:
        """Return whether a vector ID exists."""
        self._validate_vector_id(vector_id)
        return vector_id in self._records

    def contains_chunk_id(self, chunk_id: str) -> bool:
        """Return whether a chunk ID exists."""
        self._validate_chunk_id(chunk_id)
        return chunk_id in self._chunk_to_vector

    def remove(self, vector_ids: list[int]) -> int:
        """Remove vector IDs and their associated chunks."""
        if not vector_ids:
            return 0

        removed = 0

        for vector_id in vector_ids:
            self._validate_vector_id(vector_id)

            chunk = self._records.pop(vector_id, None)

            if chunk is None:
                continue

            self._chunk_to_vector.pop(chunk.chunk_id, None)
            removed += 1

        return removed

    def clear(self) -> None:
        """Remove all metadata records."""
        self._records.clear()
        self._chunk_to_vector.clear()

    def items(self) -> list[tuple[int, CodeChunk]]:
        """Return vector ID and chunk pairs in deterministic order."""
        return sorted(self._records.items(), key=lambda item: item[0])

    def chunks(self) -> list[CodeChunk]:
        """Return all chunks in vector-ID order."""
        return [chunk for _, chunk in self.items()]

    def save(self, path: str | Path) -> Path:
        """Persist metadata atomically to a JSON file."""
        destination = Path(path)

        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")

        destination.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": self.FORMAT_VERSION,
            "records": [
                {
                    "vector_id": vector_id,
                    "chunk": chunk.to_dict(),
                }
                for vector_id, chunk in self.items()
            ],
        }

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.stem}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(
                    payload,
                    temporary_file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                temporary_file.write("\n")

            os.replace(temporary_path, destination)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

            raise MetadataStoreError(
                f"Failed to save metadata to {destination}: {exc}"
            ) from exc

        return destination

    def load(self, path: str | Path) -> None:
        """Load metadata from a JSON file."""
        source = Path(path)

        if source.suffix.lower() != ".json":
            source = source.with_suffix(".json")

        if not source.is_file():
            raise VectorStoreNotFoundError(
                f"Metadata file does not exist: {source}"
            )

        try:
            with source.open("r", encoding="utf-8") as metadata_file:
                payload = json.load(metadata_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise MetadataStoreCorruptionError(
                f"Unable to read metadata file {source}: {exc}"
            ) from exc

        try:
            self._load_payload(payload)
        except (TypeError, ValueError, KeyError, VectorStoreInputError) as exc:
            raise MetadataStoreCorruptionError(
                f"Invalid metadata file {source}: {exc}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Return the complete metadata store as a dictionary."""
        return {
            "format_version": self.FORMAT_VERSION,
            "size": self.size,
            "records": [
                {
                    "vector_id": vector_id,
                    "chunk": chunk.to_dict(),
                }
                for vector_id, chunk in self.items()
            ],
        }

    def _load_payload(self, payload: Any) -> None:
        """Validate and load a serialized metadata payload."""
        if not isinstance(payload, dict):
            raise TypeError("metadata payload must be an object.")

        format_version = payload.get("format_version")

        if format_version != self.FORMAT_VERSION:
            raise ValueError(
                "Unsupported metadata format version: "
                f"{format_version}"
            )

        records = payload.get("records")

        if not isinstance(records, list):
            raise TypeError("metadata records must be a list.")

        new_records: dict[int, CodeChunk] = {}
        new_chunk_to_vector: dict[str, int] = {}

        for record in records:
            if not isinstance(record, dict):
                raise TypeError("each metadata record must be an object.")

            vector_id = record.get("vector_id")
            chunk_payload = record.get("chunk")

            self._validate_vector_id(vector_id)

            if not isinstance(chunk_payload, dict):
                raise TypeError(
                    "chunk must be an object."
                )

            chunk = self._chunk_from_dict(chunk_payload)

            if vector_id in new_records:
                raise ValueError(
                    f"duplicate vector ID: {vector_id}"
                )

            if chunk.chunk_id in new_chunk_to_vector:
                raise ValueError(
                    f"duplicate chunk ID: {chunk.chunk_id}"
                )

            new_records[vector_id] = chunk
            new_chunk_to_vector[chunk.chunk_id] = vector_id

        self._records = new_records
        self._chunk_to_vector = new_chunk_to_vector

    @staticmethod
    def _chunk_from_dict(payload: dict[str, Any]) -> CodeChunk:
        """Reconstruct a CodeChunk from its serialized representation."""
        required_fields = {
            "chunk_id",
            "source_file_id",
            "repository_id",
            "content",
            "language",
            "start_line",
            "end_line",
            "symbol_name",
            "symbol_type",
            "ast_node_type",
            "parent_symbol",
            "is_documentation",
            "trust",
            "metadata",
        }

        missing = required_fields.difference(payload)

        if missing:
            raise ValueError(
                "chunk is missing required fields: "
                + ", ".join(sorted(missing))
            )

        language = ProgrammingLanguage(payload["language"])
        trust = SourceTrust(payload["trust"])

        return CodeChunk(
            chunk_id=payload["chunk_id"],
            source_file_id=payload["source_file_id"],
            repository_id=payload["repository_id"],
            content=payload["content"],
            language=language,
            start_line=payload["start_line"],
            end_line=payload["end_line"],
            symbol_name=payload["symbol_name"],
            symbol_type=payload["symbol_type"],
            ast_node_type=payload["ast_node_type"],
            parent_symbol=payload["parent_symbol"],
            is_documentation=payload["is_documentation"],
            trust=trust,
            metadata=dict(payload["metadata"]),
        )

    @staticmethod
    def _validate_vector_id(vector_id: int) -> None:
        """Validate a FAISS-compatible integer vector ID."""
        if not isinstance(vector_id, int) or isinstance(vector_id, bool):
            raise VectorStoreInputError(
                "vector_id must be an integer."
            )

        if vector_id < 0:
            raise VectorStoreInputError(
                "vector_id must be greater than or equal to zero."
            )

    @staticmethod
    def _validate_chunk_id(chunk_id: str) -> None:
        """Validate a chunk ID."""
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise VectorStoreInputError(
                "chunk_id must be a non-empty string."
            )

    @classmethod
    def _validate_chunk(cls, chunk: CodeChunk) -> None:
        """Validate a code chunk."""
        if not isinstance(chunk, CodeChunk):
            raise VectorStoreInputError(
                "chunk must be a CodeChunk instance."
            )

        cls._validate_chunk_id(chunk.chunk_id)