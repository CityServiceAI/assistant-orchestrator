import math
from typing import List

from app.deps.litellm_client import client
from . import loader
from .categories import CategoryRecord


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))

    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def rag_search_categories(query: str, k: int = 5) -> List[CategoryRecord]:
    """
    Повертає top-k L3-категорій (CategoryRecord) для тексту запиту.
    """
    if not loader.CATEGORIES:
        raise RuntimeError("Categories not initialized. Call init_categories() first.")

    resp = client.embeddings.create(
        model=loader.EMBEDDING_MODEL,
        input=[query],
    )
    q_vec = resp.data[0].embedding

    scored: list[tuple[float, CategoryRecord]] = []
    for c in loader.CATEGORIES:
        if c.embedding is None:
            continue
        sim = _cosine(q_vec, c.embedding)
        scored.append((sim, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for sim, c in scored[:k]]
