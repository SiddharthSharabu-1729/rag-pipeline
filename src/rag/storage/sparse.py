"""Sparse lexical vector encoder for exact keyword and acronym matching (BM25 style)."""

import math
import re
from collections import Counter
from rag.domain.models import SparseVector


class SparseBM25Encoder:
    """Computes sparse token frequency vectors with Murmur/hash-based index mapping.

    Produces SparseVector entities compatible with Qdrant and inverted index engines.
    """

    def __init__(self, vocab_space: int = 2**24) -> None:
        self.vocab_space = vocab_space
        # Common English stop words to diminish in sparse scoring
        self.stop_words = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an",
            "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
            "before", "being", "below", "between", "both", "but", "by", "can't",
            "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
            "doing", "don't", "down", "during", "each", "few", "for", "from",
            "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
            "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
            "i", "if", "in", "into", "is", "isn't", "it", "its", "itself", "me",
            "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on",
            "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
            "over", "own", "same", "she", "should", "shouldn't", "so", "some", "such",
            "than", "that", "the", "their", "theirs", "them", "themselves", "then",
            "there", "these", "they", "this", "those", "through", "to", "too", "under",
            "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what",
            "when", "where", "which", "while", "who", "whom", "why", "with", "won't",
            "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves",
        }

    def _tokenize(self, text: str) -> list[str]:
        """Extract alphanumeric tokens and acronyms, preserving identifiers."""
        raw_tokens = re.findall(r"\b[\w\-_\.]+\b", text.lower())
        tokens: list[str] = []
        for t in raw_tokens:
            cleaned = t.strip("._-")
            if len(cleaned) > 1 and cleaned not in self.stop_words:
                tokens.append(cleaned)
        return tokens

    def _token_to_index(self, token: str) -> int:
        """Deterministically map a token string to a positive 32-bit integer index."""
        # Use python hash with abs and modulo within vocab_space (ensure non-zero)
        h = 0
        for char in token:
            h = (h * 31 + ord(char)) & 0xFFFFFFFF
        return (h % (self.vocab_space - 1)) + 1

    def encode(self, text: str) -> SparseVector:
        """Encode text into sparse indices and BM25-scaled term frequency values."""
        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        tf = Counter(tokens)
        doc_len = len(tokens)
        avg_doc_len = 100.0  # reference average document length
        k1 = 1.2
        b = 0.75

        index_weight_map: dict[int, float] = {}

        for token, count in tf.items():
            idx = self._token_to_index(token)
            # BM25 term frequency saturation formula: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
            bm25_tf = (count * (k1 + 1.0)) / (count + k1 * (1.0 - b + b * (doc_len / avg_doc_len)))
            # Boost capitalized, acronym-like, or alphanumeric tokens with digits
            if any(c.isdigit() for c in token) or "-" in token or "_" in token:
                bm25_tf *= 1.5

            # Accumulate in case of hash collision
            index_weight_map[idx] = index_weight_map.get(idx, 0.0) + bm25_tf

        # Sort by index for optimal inverted index posting list traversal
        sorted_pairs = sorted(index_weight_map.items(), key=lambda x: x[0])
        indices = [p[0] for p in sorted_pairs]
        values = [round(p[1], 4) for p in sorted_pairs]

        return SparseVector(indices=indices, values=values)

