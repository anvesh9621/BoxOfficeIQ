# BoxOfficeIQ — Retrieval-Augmented Generation (RAG) & AI Assistant System

> **Version:** 1.0  
> **Module:** `backend/app/assistant/`  
> **Vector Engine:** FAISS (`IndexFlatL2`)  
> **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional)  
> **Inference Engine:** OpenAI `gpt-3.5-turbo` (with deterministic fallback)

---

## Table of Contents

1. [Why RAG for Cinema Operations?](#why-rag-for-cinema-operations)
2. [End-to-End System Architecture](#end-to-end-system-architecture)
3. [Document Ingestion & Chunking](#document-ingestion--chunking)
4. [Vector Indexing with FAISS](#vector-indexing-with-faiss)
5. [Retrieval Strategy](#retrieval-strategy)
6. [SQL Data Tool & Intent Routing](#sql-data-tool--intent-routing)
7. [LLM Integration & Prompt Construction](#llm-integration--prompt-construction)
8. [Hallucination Safeguards & Reliability](#hallucination-safeguards--reliability)

---

## Why RAG for Cinema Operations?

Cinema managers and analysts frequently need answers that combine:
1. **Unstructured Knowledge:** Methodology, definitions, business logic, system capabilities, data dictionary explanations.
2. **Structured Tabular Data:** Live numbers, historical totals, top theatres, and future forecasts.

A pure Large Language Model (LLM) alone fails in enterprise cinema settings:
- **Knowledge cutoff & blind spots:** It doesn't know BoxOfficeIQ's custom schemas, metrics, or theatre IDs.
- **Hallucinations:** When asked "What was Delhi IMAX's attendance last Saturday?", vanilla LLMs invent convincing but fictional numbers.

**Retrieval-Augmented Generation (RAG)** bridges this gap:
- Questions about *concepts and rules* retrieve exact passages from technical documentation.
- Questions about *numbers and metrics* route through deterministic SQL/pandas data query tools.
- The LLM synthesizes both into an authoritative, grounded response.

---

## End-to-End System Architecture

```
User Query: "Why is lag_7 our top feature and which theatre had the highest attendance?"
                                │
                                ▼
                   ┌───────────────────────────┐
                   │   Intent Router & Parser   │
                   └─────────────┬─────────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
┌──────────────────────┐                   ┌──────────────────────┐
│  Vector Retrieval    │                   │   SQL / Data Tool    │
│  (Unstructured)      │                   │   (Structured)       │
│                      │                   │                      │
│ 1. Embed Query       │                   │ 1. Match Query Intent│
│    (MiniLM-L6-v2)    │                   │ 2. Safe pandas / SQL │
│ 2. FAISS L2 Search   │                   │    aggregation       │
│ 3. Top-3 Doc Chunks  │                   │ 3. Return verified   │
│    from docs/*.md    │                   │    statistics table  │
└──────────┬───────────┘                   └──────────┬───────────┘
           │                                           │
           └─────────────────────┬─────────────────────┘
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │    Augmented Context      │
                   │    Prompt Assembly        │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │  OpenAI GPT-3.5-turbo     │
                   │  (or Rule-Based Fallback) │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                   Grounded, Accurate User Answer
```

---

## Document Ingestion & Chunking

### Source Corpus
The ingestion pipeline scans and indexes all markdown documentation located in `d:\BoxOfficeIQ2\docs\`:
- `README.md`
- `architecture.md`
- `data-dictionary.md`
- `ml-methodology.md`
- `api-reference.md`
- `sentiment-methodology.md`
- `rag-system.md`
- `user-guide.md`

### Chunking Strategy
- **Chunk Size:** 500 characters
- **Chunk Overlap:** 50 characters
- **Splitting Strategy:** Recursive character splitting respecting markdown section boundaries (`#`, `##`, `###`), paragraph breaks (`\n\n`), and bullet points.

*Rationale:* 500 characters captures cohesive explanations (such as a feature definition or metric formula) without exceeding the embedding model's context or diluting retrieval precision.

---

## Vector Indexing with FAISS

### Embedding Model: `all-MiniLM-L6-v2`
- **Output Dimensions:** 384
- **Speed:** ~14,000 sentences/sec on GPU, ~2,000 sentences/sec on multi-core CPU.
- **Characteristics:** Sentence-transformers model mapped to a dense vector space, tuned specifically for semantic similarity and search.

### FAISS Index: `IndexFlatL2`
- Index structure: Exact Euclidean (L2) distance search (`faiss.IndexFlatL2(384)`).
- **Why Flat L2?** For corpora of thousands of chunks, exact Flat L2 search executes in < 2 milliseconds on CPU with zero recall loss (unlike approximate HNSW or IVF indices).
- **Storage:** Persisted locally as `models/faiss_index.bin` along with serialized metadata chunks (`models/doc_metadata.pkl`).

---

## Retrieval Strategy

1. **Query Encoding:** User question string is encoded using `model.encode([query])` to produce a 384-dimensional query vector.
2. **Top-K Search:** FAISS retrieves the `K = 3` closest chunk vectors.
3. **Similarity Filtering:** Chunks with an L2 distance above threshold `D_max > 1.4` (low cosine similarity) are pruned to prevent irrelevant context injection.
4. **Metadata Extraction:** Each retrieved chunk carries its originating source file name and section header for explicit source citation.

---

## SQL Data Tool & Intent Routing

To prevent arbitrary SQL injection vulnerabilities while ensuring 100% numerical accuracy, BoxOfficeIQ utilizes **Deterministic Keyword Intent Routing**:

### Safe Routing Flow
1. Query strings are scanned for key analytical intents:
   - "highest attendance", "top theatre", "most popular"
   - "average attendance", "total tickets", "revenue"
   - "forecast for tomorrow", "next week prediction"
2. When an analytical intent is matched, execution is routed to pre-compiled, parameter-checked pandas / SQLAlchemy routines.
3. **Strict Injection Prevention:** The assistant never converts untrusted user free text directly into raw SQL string statements (`SELECT * FROM " + user_input`).

---

## LLM Integration & Prompt Construction

### System Prompt Blueprint
```
You are the BoxOfficeIQ Cinema Intelligence Assistant.
Your task is to provide helpful, professional, and factually grounded advice to cinema operators.

Guidelines:
1. Base all answers strictly on the provided Context Documents and Verified Data below.
2. If the answer cannot be found in the context or data, explicitly state:
   "I do not have sufficient data in the system to answer that with certainty."
3. Never invent metrics, attendance numbers, or theatre IDs.
4. When quoting attendance numbers, specify the theatre name and timeframe.
```

### Prompt Context Injection
```
[CONTEXT DOCUMENTS]
--- Source: ml-methodology.md ---
lag_7 captures the weekly seasonal cycle directly and is typically the strongest lag feature...

[VERIFIED DATA]
Top Theatre (2024): IMAX Grand Bengaluru (T002) - 385 daily average attendance.

[USER QUESTION]
{user_query}
```

### Deterministic Fallback Mode
If no `OPENAI_API_KEY` is present or API limits are reached:
- The system automatically triggers the rule-based assistant fallback engine.
- Extracts matching keyword templates and returns direct excerpts with clean formatting.

---

## Hallucination Safeguards & Reliability

1. **Context Grounding Requirement:** Prompts penalize speculation. The model is instructed to quote or cite context references.
2. **Confidence Intervals on Projections:** When discussing forecasts, the assistant reports the 90% confidence lower and upper bounds alongside point estimates.
3. **Read-Only Database Permissions:** Even in advanced query scenarios, the backend DB connection role has read-only (`SELECT`) permissions.
4. **Audit Trail:** Every chat response returns a `sources` array indicating whether the answer was synthesized from RAG document chunks or live SQL queries.
