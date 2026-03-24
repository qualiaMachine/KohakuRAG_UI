"""Semantic Scholar API retriever for supplementing local RAG with external papers."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Sequence

import httpx
import numpy as np

from .types import ContextSnippet, NodeKind, RetrievalMatch, StoredNode


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
        timeout: float = 10.0,
        max_results: int = 5,
    ) -> None:
        key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        headers: dict[str, str] = {}
        if key:
            headers["x-api-key"] = key
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
        )
        self._max_results = max_results

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

        try:
            resp = await self._client.get(_S2_SEARCH, params=params)
            resp.raise_for_status()
        except httpx.HTTPError:
            # Network or rate-limit error — degrade gracefully
            return []

        data = resp.json().get("data", [])

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
        tasks = [self.search_papers(q, top_k=top_k) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen_ids: set[str] = set()
        papers: list[S2Paper] = []
        for result in results:
            if isinstance(result, BaseException):
                continue
            for paper in result:
                if paper.paper_id not in seen_ids:
                    seen_ids.add(paper.paper_id)
                    papers.append(paper)

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
