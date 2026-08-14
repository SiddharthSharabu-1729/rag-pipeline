# Enterprise Production-Grade End-to-End RAG Pipeline

A robust, enterprise-ready, modular Retrieval-Augmented Generation (RAG) system engineered for high faithfulness, sub-second latency, deterministic citation grounding, and full observability.

---

## 🏛️ Architectural Highlights

```mermaid
flowchart TD
    subgraph INGESTION["1. Data Ingestion & Indexing Pipeline (Async/CDC)"]
        RawDocs["Documents (Markdown, PDF, Text)"]
        Parser["Layout-Aware Parser"]
        Chunker["Hierarchical Chunker (Parent-Child)"]
        DenseModel["Dense Embeddings (OpenAI / Mock)"]
        SparseModel["Sparse Lexical Encoder (BM25)"]
        VectorStore[("Qdrant Hybrid Vector Store")]

        RawDocs --> Parser --> Chunker
        Chunker --> DenseModel --> VectorStore
        Chunker --> SparseModel --> VectorStore
    end

    subgraph RUNTIME["2. Runtime Query & Retrieval Pipeline"]
        UserQuery["User Query"]
        GuardIn["Input Guardrails & Sanitization"]
        Cache{"Semantic Cache Hit?"}
        CachedResp["Cached Response (<15ms)"]
        Router["Query Transformer & Sub-Queries"]

        UserQuery --> GuardIn --> Cache
        Cache -- "Hit (Cosine >= 0.94)" --> CachedResp
        Cache -- "Miss" --> Router

        DenseSearch["Dense Vector Search (Top-20)"]
        SparseSearch["Sparse BM25 Search (Top-20)"]
        RRF["Reciprocal Rank Fusion (RRF)"]
        Reranker["Cross-Encoder Precision Reranker"]
        LostInMiddle["Context Optimizer (Lost-in-the-Middle)"]

        Router --> DenseSearch & SparseSearch
        DenseSearch & SparseSearch --> RRF --> Reranker --> LostInMiddle

        Gate{"Quality Gate >= Threshold?"}
        Fallback["Grounded Fallback Response"]
        LLM["Streaming LLM Generator"]
        SelfRAG["Self-RAG Groundedness & Citation Verifier"]
        GuardOut["Output Guardrails (PII Masking)"]
        FinalAnswer["Streaming Response + Verified Citations"]

        LostInMiddle --> Gate
        Gate -- "Below Threshold" --> Fallback
        Gate -- "Passed" --> LLM --> SelfRAG --> GuardOut --> FinalAnswer
    end
```

### Core Production Pillars

1. **Change Data Capture (CDC) & Layout-Aware Ingestion**:
   - Computes SHA-256 content hashes to avoid redundant re-embedding.
   - Extracts page numbers, section hierarchies, and titles.
   - Preserves source attribution metadata (`doc_id`, `parent_id`, `page_number`).

2. **Hierarchical (Parent-Child) Chunking**:
   - Generates granular leaf chunks (150–200 tokens) for sharp semantic matching.
   - Links each leaf chunk to its parent context window (800–1000 tokens) so the LLM receives complete context without fragmentation.

3. **Hybrid Search (Dense + Sparse BM25)**:
   - Combines dense semantic embeddings with sparse BM25 lexical term vectors.
   - Merges results via **Reciprocal Rank Fusion (RRF)**:
     $$RRF(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_m(d)}$$
   - Prevents blind spots on exact product names, acronyms, and part numbers.

4. **Two-Stage Retrieval & Cross-Encoder Re-Ranking**:
   - Stage 1: High-recall broad search (top 20 candidates).
   - Stage 2: Deep joint cross-attention reranking (top 5 candidates).

5. **Context Window Attention (Lost-in-the-Middle Mitigation)**:
   - Rearranges candidate chunks dynamically so that the highest scoring chunks sit at the extreme beginning and end of the context window.

6. **Self-RAG / Corrective RAG (CRAG) Verification**:
   - Automated evaluation of claim-level entailment against retrieved context.
   - Validates that every bracketed citation `[Ref-X]` matches real retrieved evidence.
   - Automatically re-prompts for self-correction if ungrounded claims are detected.

7. **Semantic Caching (< 15ms Response)**:
   - Evaluates incoming query vectors against cached query embeddings using cosine similarity.
   - Returns instant responses for repeated or semantically equivalent questions, eliminating unnecessary LLM token cost.

8. **Enterprise Guardrails & PII Masking**:
   - Redacts email addresses, telephone numbers, Social Security numbers, credit cards, and secret API keys before final output delivery.

9. **Quantitative Evaluation Suite (RAG Triad)**:
   - Measures **Context Precision**, **Context Recall**, **Faithfulness**, and **Answer Relevance**.

---

## 📂 Repository Structure

```
c:\Users\siddh\OneDrive\Desktop\RAG\
├── pyproject.toml
├── uv.lock
├── README.md
├── src/
│   └── rag/
│       ├── __init__.py               # Package exports & CLI entrypoint
│       ├── config.py                 # Pydantic Settings configuration
│       ├── domain/
│       │   ├── models.py             # Domain models (Document, Chunk, Query, SearchResult, RAGResponse)
│       │   └── exceptions.py         # Custom domain exceptions
│       ├── ingestion/
│       │   ├── parsers/              # MarkdownParser, PDFParser, TextParser, ParserFactory
│       │   ├── chunkers/             # HierarchicalChunker (Parent-Child)
│       │   └── pipeline.py           # Ingestion ETL with Change Data Capture (CDC)
│       ├── storage/
│       │   ├── base.py               # Abstract BaseVectorStore & BaseEmbeddingClient
│       │   ├── embeddings.py         # MockEmbeddingClient & OpenAIEmbeddingClient
│       │   ├── sparse.py             # SparseBM25Encoder (Lexical term weighting)
│       │   ├── qdrant_store.py       # Qdrant hybrid (dense + sparse) storage engine
│       │   └── semantic_cache.py     # Sub-15ms vector semantic cache
│       ├── retrieval/
│       │   ├── query_transform.py    # Query sanitization, contextualization & sub-queries
│       │   ├── hybrid_retriever.py   # Dense + Sparse + Reciprocal Rank Fusion
│       │   ├── reranker.py           # Cross-Encoder re-ranking
│       │   └── context_optimizer.py  # Lost-in-the-middle ordering & quality gating
│       ├── generation/
│       │   ├── prompt_templates.py   # Grounded citation prompts & system instructions
│       │   ├── llm_client.py         # Async streaming LLM clients (OpenAI + Mock)
│       │   ├── self_rag_verifier.py  # Faithfulness & citation validity verification
│       │   └── guardrails.py         # PII redaction & safety filters
│       ├── pipeline/
│       │   └── rag_pipeline.py       # End-to-end async orchestrator (aquery + astream)
│       ├── evaluation/
│       │   ├── metrics.py            # RAG Triad evaluators
│       │   └── evaluator.py          # Automated benchmark runner
│       └── api/
│           ├── app.py                # FastAPI application setup
│           └── routes/
│               ├── query.py          # /api/v1/query and /api/v1/query/stream (SSE)
│               ├── ingest.py         # /api/v1/ingest/text and /api/v1/ingest/file
│               └── health.py         # /api/v1/health
└── tests/
    ├── test_ingestion.py             # Parser & HierarchicalChunker tests
    ├── test_storage.py               # Qdrant, SparseBM25, and SemanticCache tests
    ├── test_retrieval_and_reranking.py# QueryTransformer, Reranker & ContextOptimizer tests
    ├── test_guardrails_and_self_rag.py# PII & SelfRAG tests
    └── test_end_to_end.py            # Full end-to-end integration and streaming tests
```

---

## 🚀 Quickstart & Usage

### 1. Run Unit & Integration Tests
```powershell
uv run pytest -v
```

### 2. Start the REST & Streaming API
```powershell
uv run uvicorn rag.api.app:app --host 0.0.0.0 --port 8000
```
Interactive Swagger API documentation will be available at: `http://localhost:8000/docs`

### 3. Ingest a Document
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/text" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Cloud Architecture",
       "text": "The storage cluster runs with a 3-node Raft consensus group. Recovery Point Objective (RPO) is 15 minutes."
     }'
```

### 4. Query with Citations
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the RPO for the storage cluster?"}'
```

### 5. Stream Tokens in Real-Time (Server-Sent Events)
```bash
curl -N -X POST "http://localhost:8000/api/v1/query/stream" \
     -H "Content-Type: application/json" \
     -d '{"query": "How does replication work?"}'
```

---

## ⚙️ Configuration (.env)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `EMBEDDING_PROVIDER` | `mock` | `mock` (offline testing) or `openai` |
| `LLM_PROVIDER` | `mock` | `mock` (offline testing) or `openai` |
| `OPENAI_API_KEY` | None | API key for OpenAI models |
| `QDRANT_COLLECTION_NAME`| `rag_enterprise_docs` | Qdrant collection name |
| `USE_IN_MEMORY_QDRANT` | `True` | Set to `False` when pointing to remote Qdrant |
| `QDRANT_URL` | None | URL of Qdrant server (e.g. `http://localhost:6333`) |
| `SEMANTIC_CACHE_ENABLED`| `True` | Toggle sub-15ms semantic caching |
| `SEMANTIC_CACHE_THRESHOLD` | `0.94` | Cosine similarity threshold for cache hits |
| `ENABLE_PII_MASKING` | `True` | Auto-mask emails, phone numbers, and keys |
| `ENABLE_SELF_RAG_VERIFICATION` | `True` | Hallucination and citation verification |

