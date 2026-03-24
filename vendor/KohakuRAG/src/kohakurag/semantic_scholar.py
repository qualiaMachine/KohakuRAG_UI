"""Semantic Scholar API retriever for supplementing local RAG with external papers."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Sequence

import httpx
import numpy as np

from .types import ContextSnippet, NodeKind, RetrievalMatch, StoredNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class S2Paper:
    """Minimal representation of a Semantic Scholar paper."""

    paper_id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    citation_count: int = 0
    url: str = ""


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

# Semantic Scholar API base URL and endpoints
_S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_SEARCH = f"{_S2_API_BASE}/paper/search"
_S2_FIELDS = "paperId,title,abstract,authors,year,citationCount,url"


class SemanticScholarRetriever:
    """Fetches papers from the Semantic Scholar API and converts them to ContextSnippets.

    Usage::

        retriever = SemanticScholarRetriever(api_key="...")
        snippets = await retriever.search("energy consumption of LLM training", top_k=5)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 15.0,
        max_results: int = 5,
    ) -> None:
        key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        self._has_api_key = bool(key)
        headers: dict[str, str] = {}
        if key:
            headers["x-api-key"] = key
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
        )
        self._max_results = max_results
        print(
            f"[S2] SemanticScholarRetriever initialized "
            f"(api_key={'set' if self._has_api_key else 'NOT SET'}, "
            f"timeout={timeout:.0f}s, max_results={max_results})",
            flush=True,
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    async def search_papers(
        self,
        query: str,
        *,
        top_k: int | None = None,
        year_range: str | None = None,
    ) -> list[S2Paper]:
        """Search Semantic Scholar for papers matching *query*.

        Args:
            query: Natural-language search query.
            top_k: Max papers to return (defaults to ``self._max_results``).
            year_range: Optional year filter, e.g. ``"2020-2025"`` or ``"2020-"``.

        Returns:
            List of :class:`S2Paper` objects (may be shorter than *top_k*
            if fewer results are available or abstracts are missing).
        """
        k = top_k or self._max_results
        params: dict[str, str | int] = {
            "query": query,
            "limit": min(k * 2, 100),  # fetch extra to filter out abstract-less papers
            "fields": _S2_FIELDS,
        }
        if year_range:
            params["year"] = year_range

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._client.get(_S2_SEARCH, params=params)
                resp.raise_for_status()
                break  # success
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 429 and attempt < 2:
                    wait = (attempt + 1) * 2  # 2s, 4s
                    print(f"[S2] Rate-limited (429), retrying in {wait}s...", flush=True)
                    await asyncio.sleep(wait)
                    last_exc = exc
                    continue
                print(
                    f"[S2] HTTP {status} for query {query[:80]!r}: "
                    f"{exc.response.text[:200]}",
                    flush=True,
                )
                return []
            except httpx.HTTPError as exc:
                print(
                    f"[S2] Request failed for query {query[:80]!r}: {exc}",
                    flush=True,
                )
                return []
        else:
            # All retries exhausted (rate-limited)
            print(f"[S2] Rate limit retries exhausted for query {query[:80]!r}", flush=True)
            return []

        data = resp.json().get("data", [])
        print(
            f"[S2] Search returned {len(data)} raw results for query "
            f"{query[:80]!r} (limit={params['limit']})",
            flush=True,
        )

        papers: list[S2Paper] = []
        for item in data:
            abstract = item.get("abstract") or ""
            if not abstract:
                continue  # skip papers without abstracts — no useful context
            authors = [a.get("name", "") for a in (item.get("authors") or [])]
            papers.append(S2Paper(
                paper_id=item.get("paperId", ""),
                title=item.get("title", ""),
                abstract=abstract,
                authors=authors,
                year=item.get("year"),
                citation_count=item.get("citationCount", 0),
                url=item.get("url", ""),
            ))
            if len(papers) >= k:
                break

        return papers

    # ------------------------------------------------------------------
    # Multi-query search (mirrors the multi-query pattern in RAGPipeline)
    # ------------------------------------------------------------------

    async def search_multi(
        self,
        queries: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[S2Paper]:
        """Run multiple queries concurrently and deduplicate results by paper_id."""
        print(f"[S2] Searching {len(queries)} queries (top_k={top_k})...", flush=True)
        tasks = [self.search_papers(q, top_k=top_k) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen_ids: set[str] = set()
        papers: list[S2Paper] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                print(f"[S2] search_multi query {i} failed: {result}", flush=True)
                continue
            for paper in result:
                if paper.paper_id not in seen_ids:
                    seen_ids.add(paper.paper_id)
                    papers.append(paper)

        print(f"[S2] search_multi total: {len(papers)} unique papers from {len(queries)} queries", flush=True)
        return papers

    # ------------------------------------------------------------------
    # Conversion to pipeline types
    # ------------------------------------------------------------------

    @staticmethod
    def papers_to_snippets(
        papers: list[S2Paper],
        *,
        rank_offset: int = 0,
    ) -> list[ContextSnippet]:
        """Convert S2 papers into ContextSnippets for the RAG prompt.

        Each paper becomes a snippet whose text is the abstract prefixed by
        the title and author info. The ``node_id`` uses a ``s2:`` prefix so
        downstream code can distinguish external results from local ones.

        Args:
            papers: Papers from :meth:`search_papers`.
            rank_offset: Starting rank value (so S2 snippets rank after local ones).

        Returns:
            List of :class:`ContextSnippet`.
        """
        snippets: list[ContextSnippet] = []
        for i, paper in enumerate(papers):
            # Build a readable text block
            author_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                author_str += " et al."
            year_str = f" ({paper.year})" if paper.year else ""

            text = f"{paper.title}\n{author_str}{year_str}\n\n{paper.abstract}"

            # Create a ref_id that the LLM can cite
            # Use first-author-surname + year pattern when possible
            ref_id = _make_ref_id(paper)

            snippets.append(ContextSnippet(
                node_id=f"s2:{paper.paper_id}",
                document_title=paper.title,
                text=text,
                metadata={
                    "document_id": ref_id,
                    "source": "semantic_scholar",
                    "url": paper.url,
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                },
                rank=rank_offset + i,
                score=0.0,  # no embedding score for API results
            ))

        return snippets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ref_id(paper: S2Paper) -> str:
    """Create a citable ref_id like ``smith2024`` from a paper."""
    surname = ""
    if paper.authors:
        # Take last word of first author's name as surname
        parts = paper.authors[0].split()
        if parts:
            surname = parts[-1].lower()
            # Strip non-alpha characters
            surname = "".join(c for c in surname if c.isalpha())
    year = str(paper.year) if paper.year else ""
    if surname and year:
        return f"s2_{surname}{year}"
    if paper.paper_id:
        return f"s2_{paper.paper_id[:12]}"
    return "s2_unknown"
