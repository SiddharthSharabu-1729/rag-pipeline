"""Hierarchical (Parent-Child) Chunker for high-precision retrieval and rich synthesis context."""

import re
from uuid import uuid4
from rag.config import settings
from rag.domain.models import Chunk, Document
from rag.ingestion.chunkers.base import BaseChunker


class HierarchicalChunker(BaseChunker):
    """Generates granular leaf chunks linked to expansive parent context chunks."""

    def __init__(
        self,
        parent_chunk_size: int = settings.parent_chunk_size,
        parent_chunk_overlap: int = settings.parent_chunk_overlap,
        leaf_chunk_size: int = settings.leaf_chunk_size,
        leaf_chunk_overlap: int = settings.leaf_chunk_overlap,
    ) -> None:
        super().__init__()
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.leaf_chunk_size = leaf_chunk_size
        self.leaf_chunk_overlap = leaf_chunk_overlap

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text along paragraph and section boundaries."""
        raw_paras = re.split(r"\n\s*\n", text)
        cleaned = [p.strip() for p in raw_paras if p.strip()]
        return cleaned if cleaned else [text.strip()]

    def _extract_page_number(self, text: str) -> int | None:
        """Extract page number from page marker if present."""
        match = re.search(r"---\s*\[Page\s+(\d+)\]\s*---", text)
        if match:
            return int(match.group(1))
        return None

    def _build_sliding_windows(
        self,
        units: list[str],
        target_size: int,
        overlap_size: int,
        joiner: str = "\n\n",
    ) -> list[str]:
        """Group text units into sliding windows respecting token budgets."""
        windows: list[str] = []
        current_units: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)
            if current_tokens + unit_tokens > target_size and current_units:
                windows.append(joiner.join(current_units))
                # Retain overlap from end of current_units
                overlap_units: list[str] = []
                overlap_tokens = 0
                for u in reversed(current_units):
                    u_tok = self.count_tokens(u)
                    if overlap_tokens + u_tok <= overlap_size:
                        overlap_units.insert(0, u)
                        overlap_tokens += u_tok
                    else:
                        break
                current_units = overlap_units
                current_tokens = overlap_tokens

            current_units.append(unit)
            current_tokens += unit_tokens

        if current_units:
            windows.append(joiner.join(current_units))

        return windows

    def _split_text_to_leaf_windows(self, parent_text: str) -> list[str]:
        """Subdivide parent text into leaf windows."""
        sentences = re.split(r"(?<=[.!?])\s+", parent_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [parent_text]

        leaf_texts = self._build_sliding_windows(
            units=sentences,
            target_size=self.leaf_chunk_size,
            overlap_size=self.leaf_chunk_overlap,
            joiner=" ",
        )
        return leaf_texts

    def chunk(self, document: Document) -> list[Chunk]:
        """Produce hierarchical leaf chunks with attached parent context."""
        paragraphs = self._split_into_paragraphs(document.content)
        parent_texts = self._build_sliding_windows(
            units=paragraphs,
            target_size=self.parent_chunk_size,
            overlap_size=self.parent_chunk_overlap,
        )

        all_leaf_chunks: list[Chunk] = []

        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = f"{document.id}-p{p_idx:03d}"
            leaf_texts = self._split_text_to_leaf_windows(parent_text)

            for l_idx, leaf_text in enumerate(leaf_texts):
                chunk_id = f"{parent_id}-l{l_idx:03d}"
                page_num = self._extract_page_number(leaf_text) or self._extract_page_number(parent_text)

                leaf_chunk = Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.id,
                    parent_id=parent_id,
                    parent_content=parent_text,
                    content=leaf_text,
                    chunk_type="text",
                    page_number=page_num,
                    token_count=self.count_tokens(leaf_text),
                    metadata={
                        "source_uri": document.metadata.source_uri,
                        "title": document.metadata.title,
                        "acl_groups": document.metadata.acl_groups,
                        "parent_id": parent_id,
                        "leaf_index": l_idx,
                        "parent_index": p_idx,
                        **document.metadata.extra,
                    },
                )
                all_leaf_chunks.append(leaf_chunk)

        return all_leaf_chunks
