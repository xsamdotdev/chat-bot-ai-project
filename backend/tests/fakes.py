"""In-memory fakes for the Pinecone SDK — used by offline tests.

`FakePinecone` is installed over `pinecone.Pinecone` via monkeypatch, so the
production code paths (client -> index -> upsert/query/delete) are exercised
without any network access.
"""

import hashlib
import math
from types import SimpleNamespace


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class FakeIndex:
    """Minimal stand-in for `pinecone.Index` (vectors + namespaces + delete by id)."""

    def __init__(self):
        self.vectors: dict[str, tuple[list[float], dict, str]] = {}
        self.deleted: list[tuple[list[str], str]] = []

    def upsert(self, vectors, namespace=""):
        for vector_id, values, metadata in vectors:
            self.vectors[vector_id] = (list(values), dict(metadata), namespace or "")

    def query(self, vector, top_k=4, include_metadata=True, namespace="", filter=None):  # noqa: A002
        ns = namespace or ""
        scored = [
            (vector_id, _cosine(vector, values), metadata)
            for vector_id, (values, metadata, vector_ns) in self.vectors.items()
            if vector_ns == ns
        ]
        if filter:
            scored = [row for row in scored if _matches(row[2], filter)]
        scored.sort(key=lambda row: row[1], reverse=True)
        return SimpleNamespace(
            matches=[
                SimpleNamespace(id=vector_id, score=score, metadata=metadata)
                for vector_id, score, metadata in scored[:top_k]
            ]
        )

    def delete(self, ids, namespace=""):
        ns = namespace or ""
        self.deleted.append((list(ids), ns))
        for vector_id in ids:
            stored = self.vectors.get(vector_id)
            if stored and stored[2] == ns:
                del self.vectors[vector_id]

    def describe_index_stats(self):
        counts: dict[str, int] = {}
        for _values, _metadata, vector_ns in self.vectors.values():
            counts[vector_ns] = counts.get(vector_ns, 0) + 1
        return SimpleNamespace(
            total_vector_count=len(self.vectors),
            namespaces={ns: SimpleNamespace(vector_count=c) for ns, c in counts.items()},
        )


def _matches(metadata: dict, pinecone_filter: dict) -> bool:
    """Very small subset of Pinecone filter semantics: equality / $eq."""
    for key, condition in pinecone_filter.items():
        if isinstance(condition, dict):
            if "$eq" in condition and metadata.get(key) != condition["$eq"]:
                return False
        elif metadata.get(key) != condition:
            return False
    return True


class FakeInference:
    """Deterministic stand-in for `client.inference` (embed calls are recorded)."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.calls: list[dict] = []

    def embed(self, model, inputs, parameters=None):
        self.calls.append({"model": model, "inputs": list(inputs), "parameters": dict(parameters or {})})
        return [SimpleNamespace(values=self._vector(text)) for text in inputs]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [(digest[i % len(digest)] - 128) / 128.0 for i in range(self.dim)]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


class FakePinecone:
    """Stand-in for `pinecone.Pinecone`; one index instance per client."""

    instances: list["FakePinecone"] = []

    def __init__(self, api_key=None, **kwargs):
        self.api_key = api_key
        self.index_name = None
        self.index_host = None
        self.inference = FakeInference()
        self._index = FakeIndex()
        FakePinecone.instances.append(self)

    def Index(self, name=None, host=None):  # noqa: N802 - mirrors the SDK naming
        self.index_name = name
        self.index_host = host
        return self._index
