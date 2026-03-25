"""High-level RAG pipeline orchestration."""

from __future__ import annotations

import base64
import json
import re
import time as _time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping, Protocol, Sequence

from .datastore import HierarchicalNodeStore, InMemoryNodeStore, matches_to_snippets

if TYPE_CHECKING:
    from .datastore import ImageStore
from .embeddings import EmbeddingModel, JinaEmbeddingModel
from .reranker import CrossEncoderReranker
from .semantic_scholar import SemanticScholarRetriever
from .types import ContextSnippet, NodeKind, RetrievalMatch, StoredNode


# ============================================================================
# PROTOCOLS
# ============================================================================


class ChatModel(Protocol):
    """Protocol for chat backends."""

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:  # pragma: no cover
        raise NotImplementedError


class QueryPlanner(Protocol):
    """Protocol for query expansion/rewriting."""

    async def plan(self, question: str) -> Sequence[str]:  # pragma: no cover
        raise NotImplementedError


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class RetrievalResult:
    """Container for retrieval outputs."""

    question: str
    matches: list[RetrievalMatch]  # Direct vector search results
    snippets: list[ContextSnippet]  # Expanded with parent/child context
    image_nodes: list[StoredNode] | None = (
        None  # All images (for backward compatibility)
    )
    images_from_text: list[StoredNode] | None = (
        None  # Images from text retrieval (caption only)
    )
    images_from_vision: list[StoredNode] | None = (
        None  # Images from image search (send as images)
    )


@dataclass
class StructuredAnswer:
    """Structured answer format (for WattBot and similar tasks)."""

    answer: str
    answer_value: str
    ref_id: list[str]
    explanation: str
    ref_url: list[str] = field(default_factory=list)
    supporting_materials: str = ""


@dataclass
class StructuredAnswerResult:
    """Complete result from structured QA pipeline."""

    answer: StructuredAnswer
    retrieval: RetrievalResult
    raw_response: str
    prompt: str
    timing: dict[str, float] = field(default_factory=dict)
    # timing keys (seconds): retrieval_s, generation_s, total_s


@dataclass
class PromptTemplate:
    """Template for building LLM prompts with dynamic context."""

    system_prompt: str
    user_template: str  # Must have {question}, {context}, {additional_info_json}
    additional_info: Mapping[str, object] | None = None

    def render(
        self,
        *,
        question: str,
        snippets: Sequence[ContextSnippet],
        image_nodes: Sequence[StoredNode] | None = None,
    ) -> str:
        """Fill template with question and retrieved context.

        Args:
            question: User question
            snippets: Retrieved context snippets
            image_nodes: Optional image nodes from sections (for image-aware RAG)

        Returns:
            Rendered prompt string
        """
        # Format context with optional images
        if image_nodes:
            context = format_context_with_images(snippets, image_nodes)
        else:
            context = format_snippets(snippets)

        extras = self.additional_info or {}
        extras_json = json.dumps(extras, ensure_ascii=False)

        return self.user_template.format(
            question=question,
            context=context,
            additional_info_json=extras_json,
            additional_info=extras,
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def format_snippets(snippets: Sequence[ContextSnippet]) -> str:
    """Render snippets as formatted context string for LLM prompt.

    Format: [ref_id=doc node=sec1:p2 score=0.95] snippet text
    Snippets separated by --- lines for readability.
    """
    blocks: list[str] = []

    for snippet in snippets:
        meta = snippet.metadata or {}
        doc_id = str(meta.get("document_id", "unknown"))
        full_node_id = snippet.node_id

        # Compact node ID: amazon2023:sec1:p2 → sec1:p2
        node_label = (
            full_node_id.split(":", 1)[1] if ":" in full_node_id else full_node_id
        )

        # header = f"[ref_id={doc_id} node={node_label} score={snippet.score:.3f}] "
        header = f"[ref_id={doc_id}] "  # only necessary info to avoid LLM hallucination and waste tokens
        text = snippet.text.strip()
        blocks.append(header + text)

    return "\n---\n".join(blocks)


def format_image_nodes(image_nodes: Sequence[StoredNode]) -> str:
    """Format image nodes for LLM prompt.

    Format:
        [ref_id=doc1] [img:name WxH] Caption text...

        [ref_id=doc2] [img:name2 WxH2] Caption text 2...

    Returns empty string if no images.
    """
    if not image_nodes:
        return ""

    blocks: list[str] = []
    for node in image_nodes:
        # Get document ID from metadata
        doc_id = node.metadata.get("document_id", "unknown")

        # Image text is already in format: [img:name WxH] caption...
        # Add ref_id prefix to match text snippet format
        formatted = f"[ref_id={doc_id}] {node.text.strip()}"
        blocks.append(formatted)

    return "\n\n".join(blocks)


def format_context_with_images(
    snippets: Sequence[ContextSnippet],
    image_nodes: Sequence[StoredNode] | None = None,
) -> str:
    """Format context with separate sections for text and images.

    Format:
        Context snippets:
        [ref_id=doc1] Text...
        ---
        [ref_id=doc2] More text...

        Referenced media:
        [img:Fig1 800x600] Bar chart showing...

        [img:Fig2 1200x900] Diagram of system...
    """
    context = format_snippets(snippets)

    if image_nodes:
        image_text = format_image_nodes(image_nodes)
        if image_text:
            context += "\n\nReferenced media:\n" + image_text

    return context


def build_multimodal_content(
    text_content: str,
    image_nodes: Sequence[StoredNode] | None,
    image_store: ImageStore | None = None,
) -> str | list[dict]:
    """Build multimodal content for vision-capable LLMs.

    Args:
        text_content: The main text prompt
        image_nodes: Image nodes to include (must have image_storage_key in metadata)
        image_store: ImageStore instance for retrieving image bytes

    Returns:
        - If no images or no image_store: returns text_content as string
        - Otherwise: returns list of content parts for multimodal LLM
    """
    if not image_nodes or image_store is None:
        return text_content

    content_parts: list[dict] = [{"type": "text", "text": text_content}]

    for node in image_nodes:
        storage_key = node.metadata.get("image_storage_key")
        if not storage_key:
            continue

        try:
            # Use sync method directly (avoids async complexity in prompt building)
            image_bytes = image_store._sync_get(storage_key)
            if image_bytes:
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/webp;base64,{image_b64}"},
                    }
                )
        except Exception:
            continue  # Skip failed images

    return content_parts if len(content_parts) > 1 else text_content


# ============================================================================
# DEFAULT IMPLEMENTATIONS
# ============================================================================


class SimpleQueryPlanner:
    """Pass-through planner that uses the raw question without expansion."""

    async def plan(self, question: str) -> Sequence[str]:
        """Return single-element list containing the original question."""
        return [question]


_QUERY_PLANNER_PROMPT = """\
You are a search query planner for a technical document retrieval system.
Given a user question, generate {n} diverse search queries that together
cover different terminologies, synonyms, and sub-questions that would help
retrieve all relevant passages from a corpus of research papers.

Rules:
- Each query should target a different angle or terminology for the same information need.
- Include the original question (possibly lightly rephrased) as the first query.
- Use varied technical vocabulary (e.g., "energy consumption" vs "power usage" vs "electricity demand").
- If the question has sub-parts, dedicate a query to each sub-part.
- Return ONLY a JSON array of strings, no explanation.

Question: {question}

JSON array of {n} queries:"""


class LLMQueryPlanner:
    """Query planner that uses an LLM to expand a single question into diverse retrieval queries."""

    def __init__(
        self,
        chat_model: "ChatModel",
        max_queries: int = 3,
    ) -> None:
        self._chat = chat_model
        self._max_queries = max_queries

    async def plan(self, question: str) -> Sequence[str]:
        """Expand *question* into up to *max_queries* diverse retrieval queries."""
        prompt = _QUERY_PLANNER_PROMPT.format(n=self._max_queries, question=question)
        try:
            raw = await self._chat.complete(
                prompt, system_prompt="You are a helpful search query planner."
            )
            # Parse JSON array from response
            start = raw.index("[")
            end = raw.rindex("]") + 1
            queries = json.loads(raw[start:end])
            if isinstance(queries, list) and queries:
                # Ensure strings and limit to max_queries
                queries = [str(q).strip() for q in queries if str(q).strip()]
                return queries[: self._max_queries] if queries else [question]
        except Exception:
            pass
        # Fallback: return original question
        return [question]


class MockChatModel:
    """Dummy LLM for testing (returns truncated context)."""

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Extract context from prompt and return as mock answer."""
        return "Mock response:\n" + prompt.split("Context:", 1)[-1].strip()[:200]


# ============================================================================
# RAG PIPELINE
# ============================================================================


class RAGPipeline:
    """Coordinates query planning, retrieval, and LLM answering."""

    def __init__(
        self,
        *,
        store: HierarchicalNodeStore | None = None,
        embedder: EmbeddingModel | None = None,
        chat_model: ChatModel | None = None,
        planner: QueryPlanner | None = None,
        top_k: int = 5,
        deduplicate_retrieval: bool = False,
        rerank_strategy: str | None = None,
        top_k_final: int | None = None,
        image_store: ImageStore | None = None,
        no_overlap: bool = False,
        bm25_top_k: int = 0,
        cross_encoder: CrossEncoderReranker | None = None,
        semantic_scholar: SemanticScholarRetriever | None = None,
        semantic_scholar_top_k: int = 5,
    ) -> None:
        """Initialize RAG pipeline with pluggable components.

        All components default to in-memory/mock implementations for testing.

        Args:
            store: Vector database for storing and searching nodes
            embedder: Model for converting text to embeddings
            chat_model: LLM for generating answers
            planner: Query expansion strategy
            top_k: Default number of results per query
            deduplicate_retrieval: Whether to deduplicate results by node_id
            rerank_strategy: Strategy for reranking multi-query results
                            Options: None, "frequency", "score", "combined"
            top_k_final: Optional truncation after dedup+rerank (None = no truncation)
                        Example: top_k=16, max_queries=3, top_k_final=20
                        -> retrieves 48 docs, dedup+rerank, truncate to 20
            image_store: Optional ImageStore for vision-enabled LLM support
            no_overlap: If True, remove overlapping snippets during context expansion.
                       When parent-child pairs exist, only keep the parent to avoid
                       redundant text in the context.
            bm25_top_k: Number of additional results from BM25 sparse search (0 = disabled).
                       These results are added to dense retrieval results for context expansion,
                       NOT used for score fusion. This adds complementary keyword-matched content.
            cross_encoder: Optional cross-encoder reranker for improved passage scoring.
                          Applied after heuristic reranking, before top_k_final truncation.
            semantic_scholar: Optional Semantic Scholar retriever for external paper search.
                             When enabled, S2 abstracts are appended to local retrieval context.
            semantic_scholar_top_k: Max external papers to include from Semantic Scholar.
        """
        self._store = store or InMemoryNodeStore()
        self._embedder = embedder or JinaEmbeddingModel()
        self._chat = chat_model or MockChatModel()
        self._planner = planner or SimpleQueryPlanner()
        self._top_k = top_k
        self._deduplicate = deduplicate_retrieval
        self._rerank_strategy = rerank_strategy
        self._top_k_final = top_k_final
        self._image_store = image_store
        self._no_overlap = no_overlap
        self._bm25_top_k = bm25_top_k
        self._cross_encoder = cross_encoder
        self._semantic_scholar = semantic_scholar
        self._semantic_scholar_top_k = semantic_scholar_top_k

    @property
    def store(self) -> HierarchicalNodeStore:
        return self._store

    async def index_documents(self, documents: Iterable[StoredNode]) -> None:
        """Bulk insert pre-built nodes into the store."""
        await self._store.upsert_nodes(list(documents))

    def _deduplicate_matches(
        self, matches: list[RetrievalMatch]
    ) -> list[RetrievalMatch]:
        """Deduplicate matches by node_id, keeping the first occurrence.

        When the same node appears in results from multiple queries, we keep
        only the first occurrence to avoid duplicate context.

        Args:
            matches: List of retrieval matches (potentially with duplicates)

        Returns:
            Deduplicated list of matches
        """
        seen_ids: set[str] = set()
        unique_matches: list[RetrievalMatch] = []

        for match in matches:
            if match.node.node_id not in seen_ids:
                seen_ids.add(match.node.node_id)
                unique_matches.append(match)

        return unique_matches

    def _rerank_matches(
        self, matches: list[RetrievalMatch], num_queries: int
    ) -> list[RetrievalMatch]:
        """Rerank matches based on the configured strategy.

        This method aggregates duplicate nodes and ranks them using
        frequency (how many queries returned it) and total score (sum of scores).

        Strategies:
        - "frequency": Sort by (frequency, total_score) descending
        - "score": Sort by total_score only (descending)
        - "combined": Sort by weighted combination of normalized frequency and total_score

        Args:
            matches: List of retrieval matches (potentially with duplicates)
            num_queries: Number of queries used for retrieval

        Returns:
            Reranked and deduplicated list of matches
        """
        if not self._rerank_strategy or not matches:
            return matches

        strategy = self._rerank_strategy.lower()

        # Aggregate stats for each unique node
        node_stats: dict[str, dict] = {}

        for match in matches:
            node_id = match.node.node_id
            if node_id not in node_stats:
                node_stats[node_id] = {
                    "match": match,  # Keep reference to match object
                    "frequency": 0,
                    "total_score": 0.0,
                    "max_score": match.score,
                }

            node_stats[node_id]["frequency"] += 1
            node_stats[node_id]["total_score"] += match.score
            node_stats[node_id]["max_score"] = max(
                node_stats[node_id]["max_score"], match.score
            )

        # Extract unique matches with aggregated scores
        unique_matches = []
        for stats in node_stats.values():
            # Update the match object's score to reflect total_score
            match = stats["match"]
            unique_matches.append(match)

        # Sort based on strategy
        if strategy == "frequency":
            # Primary: frequency, Secondary: total_score
            unique_matches.sort(
                key=lambda m: (
                    node_stats[m.node.node_id]["frequency"],
                    node_stats[m.node.node_id]["total_score"],
                ),
                reverse=True,
            )

        elif strategy == "score":
            # Sort by total_score only
            unique_matches.sort(
                key=lambda m: node_stats[m.node.node_id]["total_score"],
                reverse=True,
            )

        elif strategy == "combined":
            # Weighted combination of normalized frequency and total_score
            max_freq = max(s["frequency"] for s in node_stats.values())
            max_total_score = max(s["total_score"] for s in node_stats.values())

            # Avoid division by zero
            max_freq = max(max_freq, 1)
            max_total_score = max(max_total_score, 0.001)

            unique_matches.sort(
                key=lambda m: (
                    0.4 * (node_stats[m.node.node_id]["frequency"] / max_freq)
                    + 0.6
                    * (node_stats[m.node.node_id]["total_score"] / max_total_score)
                ),
                reverse=True,
            )

        else:
            # Unknown strategy, return as-is (deduplicated by first occurrence)
            pass

        return unique_matches

    async def retrieve(
        self, question: str, *, top_k: int | None = None, bm25_top_k: int | None = None
    ) -> RetrievalResult:
        """Execute multi-query retrieval with hierarchical context expansion.

        For each planner-generated query, we independently search the vector
        store for the top-k matching nodes (sentences or paragraphs).

        Behavior:
        - If deduplicate_retrieval=False and rerank_strategy=None:
          Results are simply concatenated in planner order (original behavior)
        - If deduplicate_retrieval=True:
          Duplicate nodes (by node_id) are removed, keeping first occurrence
        - If rerank_strategy is set:
          Results are reranked using the specified strategy
          ("frequency", "score", or "combined") with frequency + total_score
        - If top_k_final is set:
          Results are truncated to top_k_final after dedup+rerank
        - If bm25_top_k > 0:
          Additional BM25 results are appended after dense results for context expansion.
          These are NOT fused with dense scores - they add complementary keyword matches.

        Example configurations:
        1. top_k=16, max_queries=3, deduplicate=False, rerank=None, top_k_final=None
           -> 48 results (16 * 3, with potential duplicates)

        2. top_k=16, max_queries=3, deduplicate=True, rerank=None, top_k_final=None
           -> up to 48 unique results (duplicates removed)

        3. top_k=16, max_queries=3, deduplicate=True, rerank="frequency", top_k_final=20
           -> 20 results (best ranked by frequency + total score)

        4. top_k=8, bm25_top_k=4, deduplicate=True
           -> dense results (up to 8 per query) + additional BM25 results (up to 4)

        Args:
            question: User question
            top_k: Number of results per query (uses default if None)
            bm25_top_k: Number of additional BM25 results (uses default if None)

        Returns:
            RetrievalResult with matches and expanded snippets
        """
        # Generate multiple retrieval queries (or just one if simple planner)
        queries = list(await self._planner.plan(question))
        if not queries:
            raise ValueError("Planner returned no queries.")

        # Embed all queries
        query_vectors = await self._embedder.embed(queries)
        k = top_k or self._top_k
        bm25_k = bm25_top_k if bm25_top_k is not None else self._bm25_top_k

        # Execute each query independently (dense search)
        all_matches: list[RetrievalMatch] = []
        for vector in query_vectors:
            matches = await self._store.search(
                vector,
                k=k,
                kinds={
                    NodeKind.SENTENCE,
                    NodeKind.PARAGRAPH,
                },  # Skip documents/sections
            )
            all_matches.extend(matches)

        # Apply deduplication if enabled (before reranking)
        if self._deduplicate:
            all_matches = self._deduplicate_matches(all_matches)

        # Apply reranking if strategy is configured
        # Note: reranking also deduplicates and uses frequency + total_score
        if self._rerank_strategy:
            all_matches = self._rerank_matches(all_matches, len(queries))

        # Apply cross-encoder reranking if configured (after heuristic rerank)
        if self._cross_encoder is not None:
            all_matches = self._cross_encoder.rerank(
                all_matches, question,
            )

        # Apply top_k_final truncation if configured
        if self._top_k_final is not None and self._top_k_final > 0:
            all_matches = all_matches[: self._top_k_final]

        # Add BM25 results for additional context (not fused, just appended)
        if bm25_k > 0 and hasattr(self._store, "search_bm25"):
            # Collect node_ids already in dense results
            dense_node_ids = {m.node.node_id for m in all_matches}

            # Search BM25 for each query
            bm25_matches: list[RetrievalMatch] = []
            for query in queries:
                matches = await self._store.search_bm25(
                    query,
                    k=bm25_k,
                    kinds={NodeKind.SENTENCE, NodeKind.PARAGRAPH},
                )
                bm25_matches.extend(matches)

            # Deduplicate BM25 results and exclude nodes already in dense results
            seen_bm25_ids: set[str] = set()
            for match in bm25_matches:
                node_id = match.node.node_id
                if node_id not in dense_node_ids and node_id not in seen_bm25_ids:
                    seen_bm25_ids.add(node_id)
                    all_matches.append(match)
                    # Stop if we've added enough BM25 results
                    if len(seen_bm25_ids) >= bm25_k:
                        break

        # Expand each match with hierarchical context
        snippets = await matches_to_snippets(
            all_matches,
            self._store,
            parent_depth=1,  # Include parent paragraph/section
            child_depth=1,  # Include child sentences
            no_overlap=self._no_overlap,
        )

        # Append Semantic Scholar results as additional context snippets
        s2_snippets: list[ContextSnippet] = []
        if self._semantic_scholar is not None:
            # Include the original question as a query — planner sub-queries
            # can be too specific for S2's keyword-based search.
            s2_queries = [question] + [q for q in queries if q != question]
            s2_papers = await self._semantic_scholar.search_multi(
                s2_queries, top_k=self._semantic_scholar_top_k,
            )
            if s2_papers:
                s2_snippets = SemanticScholarRetriever.papers_to_snippets(
                    s2_papers, rank_offset=len(snippets),
                )
                snippets = list(snippets) + s2_snippets

        # Joint reranking: when S2 results are mixed in, rerank ALL snippets
        # together so local and S2 compete fairly on relevance.  When S2 is
        # off, local matches are already cross-encoder-ranked at match level.
        if self._cross_encoder is not None and s2_snippets and snippets:
            all_texts = [s.text for s in snippets]
            ranked = self._cross_encoder.rerank_texts(all_texts, question)
            reranked: list[ContextSnippet] = []
            for new_rank, (orig_idx, score) in enumerate(ranked):
                s = snippets[orig_idx]
                reranked.append(ContextSnippet(
                    node_id=s.node_id,
                    document_title=s.document_title,
                    text=s.text,
                    metadata=s.metadata,
                    rank=new_rank,
                    score=score,
                ))
            snippets = reranked

        return RetrievalResult(
            question=question,
            matches=all_matches,
            snippets=snippets,
        )

    async def _extract_images_from_snippets(
        self, snippets: Sequence[ContextSnippet]
    ) -> list[StoredNode]:
        """Extract image nodes from retrieved sections.

        Looks at all sections containing retrieved snippets and collects
        their image children (paragraphs with attachment_type='image').

        Args:
            snippets: Retrieved context snippets

        Returns:
            List of image nodes
        """
        image_nodes: list[StoredNode] = []
        seen_sections: set[str] = set()

        for snippet in snippets:
            # Get section ID from node ID (format: doc:sec:p:s → doc:sec)
            parts = snippet.node_id.split(":")
            if len(parts) >= 2:
                section_id = ":".join(parts[:2])
            else:
                continue  # Not a hierarchical node

            # Skip if we already processed this section
            if section_id in seen_sections:
                continue
            seen_sections.add(section_id)

            try:
                # Get the section node
                section_node = await self._store.get_node(section_id)

                # Check all children for images
                for child_id in section_node.child_ids:
                    try:
                        child_node = await self._store.get_node(child_id)

                        # Check if this is an image node
                        if child_node.metadata.get("attachment_type") in ("image", "page_image"):
                            image_nodes.append(child_node)

                    except KeyError:
                        continue  # Child node not found

            except KeyError:
                continue  # Section node not found

        return image_nodes

    async def retrieve_with_images(
        self,
        question: str,
        *,
        top_k: int | None = None,
        top_k_images: int = 0,
        bm25_top_k: int | None = None,
    ) -> RetrievalResult:
        """Execute multi-query retrieval with image extraction.

        Image retrieval strategy:
        1. Extract images from retrieved text sections (use captions only in LLM)
        2. Additionally retrieve from image-only index if top_k_images > 0 (send as actual images to vision LLM)
        3. Combine and deduplicate for backward compatibility (image_nodes)

        Args:
            question: User question
            top_k: Number of text results per query (uses default if None)
            top_k_images: Number of ADDITIONAL images from image-only index
                         (0 = only extract from sections, >0 = also search image index)
            bm25_top_k: Number of additional BM25 results (uses default if None)

        Returns:
            RetrievalResult with separate image sources:
            - images_from_text: Images from text retrieval (use captions)
            - images_from_vision: Images from image search (send as actual images)
            - image_nodes: Combined for backward compatibility
        """
        # Standard text retrieval (with BM25 if configured)
        result = await self.retrieve(question, top_k=top_k, bm25_top_k=bm25_top_k)

        # Images from text retrieval sections (captions only)
        images_from_sections = await self._extract_images_from_snippets(result.snippets)

        # Images from dedicated image search (send as actual images)
        if top_k_images > 0:
            images_from_index = await self._retrieve_images_only(question, top_k_images)
        else:
            images_from_index = []

        # Combine and deduplicate for backward compatibility
        all_images = []
        seen_ids = set()

        # Prioritize images from sections
        for node in images_from_sections:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                all_images.append(node)

        # Add images from index
        for node in images_from_index:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                all_images.append(node)

        return RetrievalResult(
            question=result.question,
            matches=result.matches,
            snippets=result.snippets,
            image_nodes=all_images if all_images else None,  # Backward compatibility
            images_from_text=images_from_sections if images_from_sections else None,
            images_from_vision=images_from_index if images_from_index else None,
        )

    async def _retrieve_images_only(self, question: str, k: int) -> list[StoredNode]:
        """Retrieve top-k images using dedicated image-only vector index.

        Args:
            question: User question
            k: Number of images to retrieve

        Returns:
            List of image nodes (empty if image index doesn't exist)
        """
        # Check if image-only index exists
        if not hasattr(self._store, "search_images"):
            return []

        # Generate retrieval queries
        queries = list(await self._planner.plan(question))
        if not queries:
            return []

        # Embed all queries
        query_vectors = await self._embedder.embed(queries)

        # Search image-only index for each query
        all_image_matches: list[StoredNode] = []
        for vector in query_vectors:
            matches = await self._store.search_images(vector, k=k)
            all_image_matches.extend([m.node for m in matches])

        # Deduplicate by node_id (in case same image matched multiple queries)
        seen_ids = set()
        unique_images = []
        for node in all_image_matches:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                unique_images.append(node)

        return unique_images[:k]  # Limit to top-k overall

    async def answer(self, question: str) -> dict:
        """Simple QA: retrieve + prompt + generate (returns unstructured dict)."""
        retrieval = await self.retrieve(question)
        prompt = self._build_prompt(question, retrieval.snippets)
        response = await self._chat.complete(prompt)

        return {
            "question": question,
            "response": response,
            "snippets": retrieval.snippets,
        }

    async def structured_answer(
        self,
        question: str,
        prompt: PromptTemplate,
        *,
        top_k: int | None = None,
        with_images: bool = False,
        top_k_images: int = 0,
        top_k_final: int | None = None,
        send_images_to_llm: bool = False,
        bm25_top_k: int | None = None,
    ) -> StructuredAnswerResult:
        """QA with custom prompt template and structured JSON parsing.

        Args:
            question: User question
            prompt: Prompt template
            top_k: Number of text results per query
            with_images: Whether to include images from retrieved sections
            top_k_images: Number of images from image-only index (0 = extract from sections)
            top_k_final: Optional override for final result truncation
            send_images_to_llm: If True and image_store is set, send actual images
                               to vision-capable LLMs instead of just captions
            bm25_top_k: Number of additional BM25 results (uses default if None)

        Returns:
            Complete structured answer result
        """
        # Temporarily override top_k_final if provided
        original_top_k_final = self._top_k_final
        if top_k_final is not None:
            self._top_k_final = top_k_final

        # --- Retrieval phase (embedding + vector search) ---
        t0 = _time.time()
        try:
            # Use image-aware retrieval if requested
            if with_images:
                retrieval = await self.retrieve_with_images(
                    question,
                    top_k=top_k,
                    top_k_images=top_k_images,
                    bm25_top_k=bm25_top_k,
                )
            else:
                retrieval = await self.retrieve(
                    question, top_k=top_k, bm25_top_k=bm25_top_k
                )
        finally:
            # Restore original top_k_final
            self._top_k_final = original_top_k_final
        t_retrieval = _time.time() - t0

        # Render user prompt with context (and images if present as captions)
        rendered_prompt = prompt.render(
            question=question,
            snippets=retrieval.snippets,
            image_nodes=retrieval.image_nodes,
        )

        # Build multimodal content if vision support is enabled
        if send_images_to_llm and retrieval.images_from_vision and self._image_store:
            prompt_content = build_multimodal_content(
                rendered_prompt,
                retrieval.images_from_vision,
                self._image_store,
            )
        else:
            prompt_content = rendered_prompt

        # --- Generation phase (LLM inference) ---
        t1 = _time.time()
        raw = await self._chat.complete(
            prompt_content, system_prompt=prompt.system_prompt
        )
        t_generation = _time.time() - t1

        # Parse JSON structure
        parsed = self._parse_structured_response(raw)

        # Collect per-component energy reported by remote services
        embed_energy_wh = getattr(self._embedder, "last_energy_wh", 0.0)
        reranker_energy_wh = 0.0
        if self._cross_encoder is not None:
            reranker_energy_wh = getattr(self._cross_encoder, "last_energy_wh", 0.0)

        return StructuredAnswerResult(
            answer=parsed,
            retrieval=retrieval,
            raw_response=raw,
            prompt=rendered_prompt,
            timing={
                "retrieval_s": t_retrieval,
                "generation_s": t_generation,
                "total_s": t_retrieval + t_generation,
                "embed_energy_wh": embed_energy_wh,
                "reranker_energy_wh": reranker_energy_wh,
            },
        )

    async def run_qa_with_feedback(
        self,
        question: str,
        *,
        system_prompt: str,
        user_template: str,
        feedback_prompt: str,
        refinement_prompt: str,
        max_feedback_rounds: int = 3,
        additional_info: Mapping[str, object] | None = None,
        top_k: int | None = None,
        with_images: bool = False,
        top_k_images: int = 0,
        send_images_to_llm: bool = False,
        bm25_top_k: int | None = None,
        on_status: object | None = None,
    ) -> StructuredAnswerResult:
        """OpenScholar-style self-feedback loop: generate → critique → re-retrieve → refine.

        Based on Asai et al. (2024) Section 2.2: Iterative Generation with
        Retrieval-Augmented Self-Feedback.

        Steps:
          1. Retrieve context and generate initial response y0
          2. Generate self-feedback F on y0 (up to max_feedback_rounds items)
          3. For each feedback fi:
             - If it suggests missing info, generate a retrieval query
             - Re-retrieve and append new passages to context
             - Refine the response: yk = LLM(yk-1, fi, context)
          4. Return final refined response

        Args:
            question: User question
            system_prompt: System prompt for initial generation
            user_template: User template for initial generation
            feedback_prompt: Template for generating self-feedback on the response.
                           Must contain {question}, {response}, {context} placeholders.
            refinement_prompt: Template for refining response with feedback.
                             Must contain {question}, {response}, {feedback},
                             {context}, {new_context} placeholders.
            max_feedback_rounds: Maximum number of feedback iterations (default 3)
            on_status: Optional callable(str) to report status updates
            Other args: Same as run_qa
        """
        import logging as _logging
        _log = _logging.getLogger(__name__)

        def _status(msg: str) -> None:
            _log.debug(msg)
            if callable(on_status):
                on_status(msg)

        # --- Step 1: Initial retrieval + generation (y0) ---
        _status("Generating initial response...")
        initial_result = await self.run_qa(
            question,
            system_prompt=system_prompt,
            user_template=user_template,
            additional_info=additional_info,
            top_k=top_k,
            with_images=with_images,
            top_k_images=top_k_images,
            send_images_to_llm=send_images_to_llm,
            bm25_top_k=bm25_top_k,
        )
        t_retrieval = initial_result.timing.get("retrieval_s", 0)
        t_generation = initial_result.timing.get("generation_s", 0)

        # Format the initial context for feedback/refinement prompts.
        # Truncate to top snippets to avoid context overflow — the feedback
        # and refinement prompts include the context PLUS the response PLUS
        # instructions, so we need headroom.  OpenScholar uses top 10 passages
        # for multi-paper tasks (Section 4.1).
        _max_feedback_snippets = 20
        feedback_snippets = initial_result.retrieval.snippets[:_max_feedback_snippets]
        initial_context = format_snippets(feedback_snippets)
        current_response = initial_result.raw_response

        # Extract the explanation text for feedback (cleaner than raw JSON)
        current_explanation = initial_result.answer.explanation or initial_result.answer.answer or ""

        # Track all snippets across iterations
        all_snippets = list(initial_result.retrieval.snippets)

        # Check if initial retrieval already included Semantic Scholar results.
        # If so, skip re-retrieval in feedback rounds — we already have broad
        # external coverage and re-retrieval is the main latency bottleneck.
        _has_s2 = any(
            s.node_id.startswith("s2:") for s in initial_result.retrieval.snippets
        )

        # --- Step 2: Self-feedback loop ---
        feedback_log: list[dict] = []
        for round_num in range(max_feedback_rounds):
            _status(f"Generating self-feedback (round {round_num + 1}/{max_feedback_rounds})...")

            # Generate feedback on current response
            t_fb_start = _time.time()
            feedback_input = feedback_prompt.format(
                question=question,
                response=current_explanation,
                context=initial_context,
            )
            feedback_raw = await self._chat.complete(
                feedback_input, system_prompt=system_prompt,
            )
            t_generation += _time.time() - t_fb_start

            # Parse feedback — expect JSON with "feedback" list and optional "retrieval_query"
            feedback_items: list[str] = []
            retrieval_query: str | None = None
            try:
                fb_start = feedback_raw.index("{")
                fb_end = feedback_raw.rindex("}") + 1
                fb_data = json.loads(feedback_raw[fb_start:fb_end])
                raw_fb = fb_data.get("feedback", [])
                if isinstance(raw_fb, str):
                    feedback_items = [raw_fb]
                elif isinstance(raw_fb, list):
                    feedback_items = [str(f) for f in raw_fb if f]
                retrieval_query = fb_data.get("retrieval_query", "").strip() or None
                # Check for "no improvements needed" signal
                if fb_data.get("done", False) or not feedback_items:
                    _status(f"Feedback round {round_num + 1}: no further improvements needed.")
                    break
            except Exception:
                # If we can't parse feedback, treat the raw text as a single feedback item
                feedback_text = feedback_raw.strip()
                if not feedback_text or "no improvement" in feedback_text.lower():
                    break
                feedback_items = [feedback_text]

            feedback_log.append({
                "round": round_num + 1,
                "feedback": feedback_items,
                "retrieval_query": retrieval_query,
            })
            _log.debug(f"Feedback round {round_num + 1}: {len(feedback_items)} items, query={retrieval_query}")

            # --- Step 3: Optional re-retrieval based on feedback ---
            # Skip if initial retrieval already has S2 results (broad coverage
            # already present — re-retrieval is the main latency bottleneck).
            new_context = ""
            if retrieval_query and not _has_s2:
                _status(f"Re-retrieving for: {retrieval_query[:80]}...")
                t_retr_start = _time.time()
                try:
                    supplemental = await self.retrieve(
                        retrieval_query,
                        top_k=top_k,
                        bm25_top_k=bm25_top_k,
                    )
                    # Deduplicate against existing snippets, limit to 10 new ones
                    existing_ids = {s.node_id for s in all_snippets}
                    new_snippets = [s for s in supplemental.snippets if s.node_id not in existing_ids][:10]
                    if new_snippets:
                        all_snippets.extend(new_snippets)
                        new_context = format_snippets(new_snippets)
                        _log.debug(f"Re-retrieval added {len(new_snippets)} new snippets")
                except Exception as e:
                    _log.warning(f"Re-retrieval failed: {e}")
                t_retrieval += _time.time() - t_retr_start
            elif retrieval_query and _has_s2:
                _log.debug("Skipping re-retrieval: initial retrieval already includes S2 papers")

            # --- Step 4: Refine response incorporating feedback ---
            _status(f"Refining response (round {round_num + 1})...")
            t_ref_start = _time.time()
            combined_feedback = "\n".join(f"- {fb}" for fb in feedback_items)
            refinement_input = refinement_prompt.format(
                question=question,
                response=current_explanation,
                feedback=combined_feedback,
                context=initial_context,
                new_context=new_context if new_context else "(no additional context)",
            )
            refined_raw = await self._chat.complete(
                refinement_input, system_prompt=system_prompt,
            )
            t_generation += _time.time() - t_ref_start

            # Parse the refined response
            refined_parsed = self._parse_structured_response(refined_raw)

            # Update current state for next iteration
            current_response = refined_raw
            current_explanation = refined_parsed.explanation or refined_parsed.answer or current_explanation

        # --- Final result assembly ---
        final_parsed = self._parse_structured_response(current_response)

        # Rebuild retrieval result with all accumulated snippets
        final_retrieval = RetrievalResult(
            question=question,
            matches=initial_result.retrieval.matches,
            snippets=all_snippets,
            image_nodes=initial_result.retrieval.image_nodes,
            images_from_text=initial_result.retrieval.images_from_text,
            images_from_vision=initial_result.retrieval.images_from_vision,
        )

        return StructuredAnswerResult(
            answer=final_parsed,
            retrieval=final_retrieval,
            raw_response=current_response,
            prompt=initial_result.prompt,
            timing={
                "retrieval_s": t_retrieval,
                "generation_s": t_generation,
                "total_s": t_retrieval + t_generation,
                "feedback_rounds": len(feedback_log),
            },
        )

    async def verify_citations(
        self,
        result: StructuredAnswerResult,
        citation_prompt: str,
    ) -> StructuredAnswerResult:
        """Lightweight post-hoc citation verification (OpenScholar Section 2.2 step 3).

        Only rewrites the explanation if it lacks inline [ref_id] citations.
        Skips the LLM call entirely if citations are already present.

        Args:
            result: The original answer result to verify.
            citation_prompt: Prompt template with {explanation} and {sources} placeholders.

        Returns:
            Updated result with citations inserted, or original if already cited.
        """
        explanation = result.answer.explanation or ""

        # Quick check: does the explanation already contain [ref_id] citations?
        if re.search(r"\[[a-z][a-z0-9_]+\d{4}", explanation):
            return result  # Already has inline citations, skip

        # Build a compact source list from top snippets
        sources = []
        for s in result.retrieval.snippets[:10]:
            doc_id = (s.metadata or {}).get("document_id", "unknown")
            sources.append(f"[ref_id={doc_id}] {s.text[:200]}")
        if not sources:
            return result

        source_text = "\n---\n".join(sources)
        prompt = citation_prompt.format(
            explanation=explanation,
            sources=source_text,
        )

        t0 = _time.time()
        raw = await self._chat.complete(prompt)
        t_verify = _time.time() - t0

        # Try to extract just the rewritten explanation
        # The LLM should return the explanation with citations inserted
        rewritten = raw.strip()

        # If the LLM wrapped it in JSON, extract the explanation field
        try:
            start = rewritten.index("{")
            end = rewritten.rindex("}") + 1
            data = json.loads(rewritten[start:end])
            if "explanation" in data:
                rewritten = str(data["explanation"]).strip()
        except Exception:
            pass

        # Validate: only use if it actually has citations now
        if not re.search(r"\[[a-z][a-z0-9_]+\d{4}", rewritten):
            return result  # Rewrite didn't help, keep original

        # Update the answer with the citation-enhanced explanation
        updated_answer = StructuredAnswer(
            answer=result.answer.answer,
            answer_value=result.answer.answer_value,
            ref_id=result.answer.ref_id,
            explanation=rewritten,
            ref_url=result.answer.ref_url,
            supporting_materials=result.answer.supporting_materials,
        )

        # Update timing
        updated_timing = dict(result.timing)
        updated_timing["generation_s"] = updated_timing.get("generation_s", 0) + t_verify
        updated_timing["total_s"] = updated_timing.get("total_s", 0) + t_verify
        updated_timing["citation_verify_s"] = t_verify

        return StructuredAnswerResult(
            answer=updated_answer,
            retrieval=result.retrieval,
            raw_response=result.raw_response,
            prompt=result.prompt,
            timing=updated_timing,
        )

    async def run_qa(
        self,
        question: str,
        *,
        system_prompt: str,
        user_template: str,
        additional_info: Mapping[str, object] | None = None,
        top_k: int | None = None,
        with_images: bool = False,
        top_k_images: int = 0,
        top_k_final: int | None = None,
        send_images_to_llm: bool = False,
        bm25_top_k: int | None = None,
    ) -> StructuredAnswerResult:
        """High-level entry point for structured question answering.

        This method keeps the RAG core generic by requiring callers to supply
        their own system prompt, user prompt template, and any per-call metadata
        via ``additional_info``.

        Args:
            question: User question
            system_prompt: System prompt for LLM
            user_template: User prompt template
            additional_info: Extra metadata for template
            top_k: Number of text results per query
            with_images: Whether to include images from retrieved sections
            top_k_images: Number of images from image-only index (requires wattbot_build_image_index.py)
            top_k_final: Optional override for final result truncation
            send_images_to_llm: If True, send actual images to vision-capable LLMs
            bm25_top_k: Number of additional BM25 results (uses pipeline default if None)

        Returns:
            Structured answer result
        """
        template = PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            additional_info=additional_info,
        )
        return await self.structured_answer(
            question=question,
            prompt=template,
            top_k=top_k,
            with_images=with_images,
            top_k_images=top_k_images,
            top_k_final=top_k_final,
            send_images_to_llm=send_images_to_llm,
            bm25_top_k=bm25_top_k,
        )

    def _build_prompt(
        self,
        question: str,
        snippets: Sequence[ContextSnippet],
    ) -> str:
        """Build simple prompt for answer() method."""
        context_blocks = []
        for snippet in snippets:
            context_blocks.append(
                f"[{snippet.document_title} | node={snippet.node_id} | score={snippet.score:.3f}]\n{snippet.text}"
            )

        context_text = "\n\n".join(context_blocks) if context_blocks else "None"

        return (
            "You are an assistant.\n"
            "Use only the provided context to answer the question.\n"
            "If the context is insufficient, respond with 'NOT ENOUGH DATA'.\n\n"
            f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer:"
        )

    @staticmethod
    def _parse_bullet_format(raw: str) -> dict | None:
        """Fallback parser for ``- key   value`` format.

        Some LLMs mimic the prompt's key-description layout instead of
        producing JSON.  This handles lines like::

            - explanation          According to [wu2021], ...
            - answer_value         1
            - ref_id               ["wu2021"]
        """
        data: dict[str, object] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            line = line[2:]
            # Split on first run of 2+ whitespace chars (key  value)
            parts = re.split(r"\s{2,}", line, maxsplit=1)
            if len(parts) != 2:
                continue
            key, value = parts[0].strip(), parts[1].strip()
            # Try to interpret value as JSON (handles lists, numbers, etc.)
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
            data[key] = value
        return data if data else None

    def _parse_structured_response(self, raw: str) -> StructuredAnswer:
        """Extract JSON from LLM response and validate fields."""
        data = None
        try:
            # Find JSON block in response
            start = raw.index("{")
            end = raw.rindex("}") + 1
            snippet = raw[start:end]
            data = json.loads(snippet)
        except Exception:
            pass

        # Fallback: try bullet-list format (- key   value)
        if data is None:
            data = self._parse_bullet_format(raw)

        if data is None:
            # Return empty structure if all parsing fails
            return StructuredAnswer(
                answer="",
                answer_value="",
                ref_id=[],
                explanation="",
            )

        # Extract and normalize fields
        answer = str(data.get("answer", "")).strip()
        answer_value = str(data.get("answer_value", "")).strip()
        explanation = str(data.get("explanation", "")).strip()

        # Parse ref_id (can be string or list)
        ref_ids_raw = data.get("ref_id", [])
        ref_ids: list[str] = []

        if isinstance(ref_ids_raw, str):
            ref_ids_raw = [ref_ids_raw]

        if isinstance(ref_ids_raw, Sequence):
            for item in ref_ids_raw:
                text = str(item).strip()
                if text:
                    # Clean up common LLM mistakes: strip "ref_id=" prefix
                    # LLM sometimes copies the context format like [ref_id=doc1]
                    if text.lower().startswith("ref_id="):
                        text = text[7:].strip()  # Remove "ref_id=" prefix
                    ref_ids.append(text)

        # Parse ref_url (can be string or list)
        ref_url_raw = data.get("ref_url", [])
        ref_urls: list[str] = []
        if isinstance(ref_url_raw, str):
            if ref_url_raw.strip() and ref_url_raw.strip() != "is_blank":
                ref_urls = [ref_url_raw.strip()]
        elif isinstance(ref_url_raw, Sequence):
            for item in ref_url_raw:
                text = str(item).strip()
                if text and text != "is_blank":
                    ref_urls.append(text)

        supporting_materials = str(data.get("supporting_materials", "")).strip()
        if supporting_materials == "is_blank":
            supporting_materials = ""

        return StructuredAnswer(
            answer=answer,
            answer_value=answer_value,
            ref_id=ref_ids,
            explanation=explanation,
            ref_url=ref_urls,
            supporting_materials=supporting_materials,
        )
