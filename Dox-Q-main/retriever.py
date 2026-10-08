"""
retriever.py — Vector retrieval for doXQ Healthcare RAG.

Fetches embeddings from MongoDB for selected documents, computes cosine
similarity in Python/numpy, and returns the top-k relevant chunks.
"""

import numpy as np
import re
from database import get_embeddings_for_documents
from embedding_model import embeddings

DEFAULT_TOP_K = 10

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9._%-]*", re.IGNORECASE)
_STOP_WORDS = {
    "about", "after", "also", "and", "are", "authors", "did", "does", "for",
    "from", "had", "has", "have", "how", "in", "into", "main", "many", "more",
    "of", "on", "patients", "say", "study", "that", "the", "their", "there",
    "these", "this", "to", "used", "were", "what", "which", "why", "with",
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("‐", "-")
        .replace("‑", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


def _query_terms(query: str) -> set[str]:
    terms = set()
    for term in _WORD_RE.findall(_normalize_text(query)):
        if len(term) < 3 or term in _STOP_WORDS:
            continue
        terms.add(term)
        terms.update(part for part in re.split(r"[-_/]", term) if len(part) >= 3 and part not in _STOP_WORDS)
    return terms


def _keyword_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0

    normalized = _normalize_text(text)
    text_terms = set(_WORD_RE.findall(normalized))
    score = len(query_terms & text_terms)

    for term in query_terms:
        count = normalized.count(term)
        if count:
            score += min(count, 3) * 0.75

    return score


def get_relevant_chunks(query: str, document_ids: list[int], k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Retrieve the top-k most relevant document chunks for a given query,
    searching ONLY within the specified document IDs.

    Parameters:
        query (str): The user's question or search string.
        document_ids (list[int]): IDs of documents to search within.
        k (int): Number of top chunks to return. Default is 10.

    Returns:
        list[dict]: Each item has 'content', 'score', 'page_number', 'document_id'.
    """
    if not document_ids:
        print("No document IDs provided for retrieval.")
        return []

    # 1. Generate embedding for the query. If the local model is unavailable,
    # still fall back to keyword scoring so exact fact questions can work.
    try:
        query_vector = np.array(embeddings.embed_query(query), dtype=np.float32)
    except Exception as e:
        print(f"Query embedding failed; falling back to keyword search: {e}")
        query_vector = None

    # 2. Fetch all embeddings for the selected documents from MongoDB
    all_embeddings = get_embeddings_for_documents(document_ids)

    if not all_embeddings:
        print("No embeddings found in MongoDB for the given document IDs.")
        return []

    print(f"Searching through {len(all_embeddings)} chunks across {len(document_ids)} document(s)...")

    # 3. Combine semantic similarity with a small lexical boost. The lexical
    # boost helps with tables, acronyms, exact numbers, and terms like SPM12.
    query_terms = _query_terms(query)
    scored = []
    for emb in all_embeddings:
        vector_score = _cosine_similarity(query_vector, emb["embedding"]) if query_vector is not None else 0.0
        keyword_score = _keyword_score(query_terms, emb["chunk_text"])
        scored.append({
            "content": emb["chunk_text"],
            "score": vector_score,
            "keyword_score": keyword_score,
            "page_number": emb["page_number"],
            "document_id": emb["document_id"],
        })

    max_keyword_score = max((item["keyword_score"] for item in scored), default=0.0)
    for item in scored:
        keyword_boost = item["keyword_score"] / max_keyword_score if max_keyword_score else 0.0
        item["rank_score"] = item["score"] + (0.25 * keyword_boost)

    # 4. Sort by combined score and return top-k
    scored.sort(key=lambda x: x["rank_score"], reverse=True)
    top_k = scored[:k]

    print(f"Retrieved {len(top_k)} relevant chunks (top score: {top_k[0]['rank_score']:.4f})" if top_k else "")

    return top_k


def get_context_string(query: str, document_ids: list[int], k: int = DEFAULT_TOP_K) -> str:
    """
    Convenience function — returns retrieved chunks joined as a single context string.
    Pass this directly to your LLM prompt.

    Parameters:
        query (str): The user's question.
        document_ids (list[int]): IDs of documents to search within.
        k (int): Number of chunks to retrieve.

    Returns:
        str: Combined context text, or empty string if nothing found.
    """
    chunks = get_relevant_chunks(query, document_ids, k=k)
    if not chunks:
        return ""
    return "\n\n---\n\n".join(
        [f"[Document {chunk['document_id']}, page {chunk['page_number']}]\n{chunk['content']}" for chunk in chunks]
    )
