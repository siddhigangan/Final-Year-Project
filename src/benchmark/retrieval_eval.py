"""Retrieval-stage poisoning experiment.

Question: if an attacker adds a poisoned variant of a snippet to the
knowledge base, does it get retrieved (and ranked above the clean
original) for the query the snippet answers?

Knowledge base = real functions from a code tree (default: this
project's own ``src/``) + the clean seed snippets + ONE poisoned
variant per scenario. Uses the real ``Retriever`` and ``FAISSVectorStore``.

Limitation: a poisoned variant is a near-duplicate of its clean seed, so
retrieval is easy by construction. Results bound the risk from
duplicate-style injection only.
"""

from __future__ import annotations

import ast
import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.benchmark.dataset import BenchmarkDataset
from src.embeddings.base import EmbeddingProvider
from src.models import CodeChunk, ProgrammingLanguage
from src.retrieval.retriever import Retriever
from src.vectorstore.faiss_store import FAISSVectorStore


class HashingEmbeddingProvider(EmbeddingProvider):
    """Offline lexical (bag-of-words hashing) embedder.

    Deterministic, no download. It matches on shared tokens, not
    meaning, so treat results as a lexical baseline, not semantic.
    """

    def __init__(self, dimension: int = 512) -> None:
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return f"hashing-bow-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z_]+", text.lower()):
                out[row, zlib.crc32(token.encode()) % self._dimension] += 1.0
            norm = np.linalg.norm(out[row])
            if norm == 0:
                out[row, 0] = 1.0
            else:
                out[row] /= norm
        return out


def collect_distractors(
    root: Path, *, limit: int = 60, per_file: int = 3
) -> list[CodeChunk]:
    """Real top-level functions from ``root`` as clean knowledge-base chunks."""
    chunks: list[CodeChunk] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        taken = 0
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or taken >= per_file:
                continue
            text = ast.get_source_segment(source, node) or ""
            if not 15 <= len(text) <= 1500:
                continue
            taken += 1
            chunks.append(CodeChunk(
                chunk_id=f"distractor-{path.name}-{node.name}",
                source_file_id=str(path), repository_id="kb-distractors",
                content=text, language=ProgrammingLanguage.PYTHON,
                start_line=node.lineno, end_line=node.end_lineno or node.lineno,
            ))
            if len(chunks) >= limit:
                return chunks
    return chunks


def _chunk(chunk_id: str, code: str, poisoned: bool) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id, source_file_id=f"{chunk_id}-file",
        repository_id="kb-seeds", content=code,
        language=ProgrammingLanguage.PYTHON, start_line=1,
        end_line=len(code.splitlines()) or 1,
        metadata={"poisoned": poisoned},
    )


@dataclass(frozen=True)
class RetrievalOutcome:
    sample_id: str
    category: str
    poison_rank: int | None
    clean_rank: int | None

    @property
    def poison_in_top_k(self) -> bool:
        return self.poison_rank is not None

    @property
    def poison_above_clean(self) -> bool:
        if self.poison_rank is None:
            return False
        return self.clean_rank is None or self.poison_rank < self.clean_rank


def evaluate_retrieval(
    dataset: BenchmarkDataset,
    *,
    provider: EmbeddingProvider,
    distractors: Sequence[CodeChunk],
    top_k: int = 5,
) -> list[RetrievalOutcome]:
    """Retrieve for each sample's query with its poisoned variant in the KB."""
    clean: dict[str, CodeChunk] = {}
    poisoned: dict[str, CodeChunk] = {}
    for s in dataset.samples:
        seed = s.metadata["seed_sample_id"]
        clean.setdefault(seed, _chunk(f"{seed}-clean", s.clean_code, False))
        poisoned[s.sample_id] = _chunk(f"{s.sample_id}-poisoned", s.poisoned_code, True)

    base = list(distractors) + list(clean.values())
    everything = base + list(poisoned.values())
    vectors = dict(zip(
        (c.chunk_id for c in everything),
        provider.embed_texts([c.content for c in everything]),
    ))

    outcomes = []
    for s in dataset.samples:
        seed = s.metadata["seed_sample_id"]
        chunks = base + [poisoned[s.sample_id]]
        store = FAISSVectorStore(provider.dimension)
        store.add(np.vstack([vectors[c.chunk_id] for c in chunks]), chunks)
        response = Retriever(provider, store).retrieve(s.query, top_k=top_k)
        ranks = {r.chunk.chunk_id: i for i, r in enumerate(response.results, 1)}
        outcomes.append(RetrievalOutcome(
            s.sample_id, s.poisoning_category,
            ranks.get(poisoned[s.sample_id].chunk_id),
            ranks.get(clean[seed].chunk_id),
        ))
    return outcomes


def summarize_retrieval(outcomes: list[RetrievalOutcome]) -> dict[str, dict]:
    buckets: dict[str, list[RetrievalOutcome]] = {}
    for o in outcomes:
        buckets.setdefault(o.category, []).append(o)
    out = {}
    for cat, rows in sorted(buckets.items()):
        found = [r.poison_rank for r in rows if r.poison_rank is not None]
        out[cat] = {
            "samples": len(rows),
            "poison_in_top_k_rate": len(found) / len(rows),
            "poison_above_clean_rate": sum(r.poison_above_clean for r in rows) / len(rows),
            "mean_poison_rank": sum(found) / len(found) if found else None,
        }
    return out
