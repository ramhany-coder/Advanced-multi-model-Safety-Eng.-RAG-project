# Enterprise Multimodal RAG Engine for OSHA 29 CFR Part 1926 (Construction Safety)

[![Architecture: LangGraph](https://img.shields.io/badge/Architecture-LangGraph-orange.svg)]()
[![Framework: LangChain%200.3](https://img.shields.io/badge/Framework-LangChain%200.3-blue.svg)]()
[![Security: Presidio%20PII%20Redaction](https://img.shields.io/badge/Security-PII%20Redaction-green.svg)]()
[![Database: Chroma%20%2B%20BM25](https://img.shields.io/badge/VectorStore-Chroma%20%26%20BM25-red.svg)]()

An enterprise-grade, highly robust, and safety-critical Multimodal Retrieval-Augmented Generation (RAG) pipeline designed to automate construction site hazard assessments and verify compliance with **OSHA 29 CFR Part 1926** standards.

This repository demonstrates production-ready AI engineering practices, focusing on **stateful multi-agent orchestration, hybrid semantic/lexical retrieval pipelines, strict PII data guardrails, and determinisgiREADME.metic caching** for cost and latency optimization.

---

## 🎯 Enterprise Use Case Ready
In industrial and construction environments, field engineers and safety managers inspect complex job sites daily. Manual compliance auditing against thousands of legal clauses is slow, tedious, and prone to oversight. 

**This engine acts as a production-ready compliance co-pilot:**
1. **Multimodal Input Handling:** An engineer uploads a site photograph (e.g., a trench profile, scaffolding setup, or fall protection rigging) alongside a natural language query.
2. **Guaranteed Data Sanitization:** The engineering layer scrubs sensitive data (faces, names, identifiable markers) before it hits external LLM APIs using text and image redaction engines.
3. **High-Fidelity Legal Verification:** The system unifies sparse keyword matching and dense vector embeddings to fetch localized chunks from OSHA regulations. If the database lacks recent regulatory changes, an autonomous agent dynamically triggers an authenticated web scraping engine.
4. **Authoritative Compliance Reports:** Synthesizes professional-grade safety audits mapped explicitly back to official standard numbers.

---

## 🏗️ Architectural Blueprint & Multi-Agent Flow

The entire system utilizes a stateful **LangGraph** architecture to govern data state transitions cleanly, preventing agent loops, maximizing audit logging accuracy, and ensuring deterministic verification before synthesizing final reports.
[ Multimodal User Query ] (Image + Text)
                         │
                         ▼
           ┌───────────────────────────┐
           │    PII Guardrail Node     │ ──► (Scrubs Text PII & Redacts Image Buffers)
           └───────────────────────────┘
                         │
                         ▼
           ┌───────────────────────────┐
           │    Semantic Cache Look    │ ──► [Hit] ──► (Instant Return Cached Response)
           └───────────────────────────┘
                         │ [Miss]
                         ▼
           ┌───────────────────────────┐
           │    Query Rewrite Agent    │ ──► (Maps informal slang to precise legal terms)
           └───────────────────────────┘
                         │
                         ▼
           ┌───────────────────────────┐
           │  Dynamic Router Engine    │
           └───────────────────────────┘
             /                       \
    [Local Knowledge]             [Web Fallback]
           /                           \
┌─────────────────────────┐     ┌──────────────────────────┐
│ Hybrid Retrieval Node   │     │  Tavily Scraper Agent    │
│ (Dense Chroma + Sparse  │     │  (Real-time Legislative  │
│  BM25 + Reranking)      │     │   Updates & Bulletins)   │
└─────────────────────────┘     └──────────────────────────┘
\                       /
\                     /
▼                   ▼
┌───────────────────────────┐
│  Response Synthesis Node  │
└───────────────────────────┘
│
▼
┌───────────────────────────┐
│   QA Audit Ranker Node    │ ──► (Validates Output Against Hallucinations)
└───────────────────────────┘
│
▼
[ Authoritative Compliance Output ]

---

## 💡 Advanced Engineering & Problem-Solving Highlights

### 1. Robust Metaprogramming for Dependency Stability
* **The Challenge:** Upstream indexing extensions (such as `bm25-retriever`) frequently rely on legacy imports (`from langchain.retrievers import ...`), which break and cause runtime failures following major framework shifts to LangChain 0.3+.
* **The Production Solution:** Rather than freezing dependencies or manually editing open-source modules inside virtual environments, an elegant metaprogramming module alias is injected at the entry point of the application lifecycle. This fixes the import path dynamically before third-party libraries load:

```python
# --- METAPROGRAMMING PATCH FOR LANGCHAIN 0.3+ COMPATIBILITY ---
import sys
import langchain_community.retrievers as community_retrievers
sys.modules['langchain.retrievers'] = community_retrievers

### Parent-Child Multi-Vector Ingestion & Hybrid Search
To avoid losing structural legal context during document fragmentation, the ingestion framework splits regulations into a dual-layered hierarchy:
* **Parents:** Full legislative articles kept completely intact within local file system storage mapping to ensure the synthesis model sees absolute context.
* **Children:** Highly granular, localized text segments split via `RecursiveCharacterTextSplitter` and converted into vector representations for high-accuracy embedding alignment.
* **Hybrid Retrieval Strategy:** Merges dense vector mathematical similarity (`ChromaDB` backed by `sentence-transformers/all-MiniLM-L6-v2`) and sparse keyword frequency matrices via a persistent `PersistentBM25Retriever`. Chunks are co-indexed, mapped back to parent documents with a dedicated unique identification tag (`doc_id`), and combined to create complete semantic and lexical coverage.

### Strict Production Data Guardrails (PII Engine)
To protect corporate liability and handle sensitive visual fields on site inspections, the pipeline implements dedicated pre-processing security steps:
* Text inputs are routed through Microsoft Presidio's `AnalyzerEngine` and `AnonymizerEngine` to scrub names, telephone numbers, and addresses.
* Image bytes undergo bounding-box metadata masking using Microsoft Presidio's `ImageRedactorEngine` to blur out structural details, text anomalies, or identification segments prior to being consumed by downstream multimodal model interfaces.

### Low-Latency Semantic Caching
To optimize operational token costs and maintain low latency under high concurrency, an advanced semantic caching layer utilizing `gptcache` intercepts queries post-sanitization. Exact or structurally parallel safety queries pull instantly from memory instead of triggering a full multi-stage agent pipeline computation.

---

## 🛠️ Codebase Architecture
The project enforces absolute separation of concerns across production-ready modules:
* **`models.py`**: Defines runtime constraints, type configurations, and schemas using `Pydantic` and typed dictionaries (`TypedDict`) to validate state maps across the entire graph.
* **`chuncking.py`**: Handles industrial data ingestion. Parses raw legislative documents, sets up structural document pairs, instantiates persistent vector layers, and exports sparse index states to physical media.
* **`agents.py`**: Houses autonomous nodes including the PII anonymizer engines, hybrid vector calculators, fallback web scrapers, and structured validation rankers.
* **`prompt.py`**: Centralizes prompt management, setting up explicit system guidelines that command models to speak in structural legal/engineering syntax while penalizing factual hallucination.

---

## ⚙️ Quickstart & Local Deployment

### 1. Installation & Environment Configuration
```bash
# Clone the repository
git clone [https://github.com/yourusername/osha-rag-engine.git](https://github.com/yourusername/osha-rag-engine.git)
cd osha-rag-engine

# Initialize a clean virtual environment
python -m venv rag_env
source rag_env/bin/activate  # On Windows use: .\rag_env\Scripts\activate

# Install requirements
pip install -r requirements.txt

2. Run the Data Ingestion Pipeline
Build your local vector databases and sparse index stores by feeding the raw scraped OSHA documents into the parsing engine:

Bash
python chuncking.py

Expected Artifact Generation:

./osha/ — Local persistent directory for dense Chroma embedding spaces.

./osha_sparse/ — Serialized token frequencies for lexical BM25 matching.

parent_doc_store_backup.json — A lightweight, high-speed key-value mapping to resolve child nodes back into their overarching regulatory contexts.

### 3. Initiate the Agent Engine
To execute queries through the entire multi-agent LangGraph system:

Bash
python main.py

🔒 Enterprise Security & Compliance Design
This system is architected for zero-trust computing environments:

Data Isolation: All vector references, raw text maps, and BM25 matrices are persisted locally or within private network boundaries.

Deterministic Evaluation: The final output node (Ranker Agent) operates as a localized quality control supervisor, matching generated response parameters back against source index metadata identifiers to ensure 0% legislative hallucination tolerance.