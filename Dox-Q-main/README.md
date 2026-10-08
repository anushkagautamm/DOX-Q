# 🧠 doXQ – Metadata-Aware RAG System for Document Understanding

**doX-Q** is an AI-powered document understanding system designed to process unstructured text in PDFs and other formats using **Large Language Models (LLMs)**. It enables users to upload documents, ask natural language questions, and receive structured, explainable responses based on the content of those documents.  
It leverages **Retrieval-Augmented Generation (RAG)** with metadata-aware embeddings to find relevant clauses and generate answers with justifications — making it suitable for sensitive, rule-based domains like **insurance**, **legal compliance**, and **contract management**.

---

## 🚀 Features

- **📂 Multi-format Document Ingestion**  
  Supports PDFs, DOCX, TXT, HTML, and email formats.

- **🔍 Metadata Extraction Pipeline**  
  Automatically extracts **titles, sample QAs**, and other contextual metadata.

- **💡 Metadata-Enhanced Embeddings**  
  Metadata is embedded alongside the main document content to improve semantic search accuracy.

- **☁️ MongoDB Atlas Vector Store**  
  Embeddings, users, documents, and chat history are stored in **MongoDB Atlas** (configure `MONGODB_URI` in `.env`).

- **🧾 Clause-Level Citation**  
  Every answer is linked back to the **exact source fragment** for transparency and auditability.

---


## 🏗 Architecture Overview

       ┌────────────────────────────┐
       │       Document Parser      │
       └─────────────┬──────────────┘
                     ▼
       ┌────────────────────────────┐
       │  Metadata Extraction Layer │
       │    (Title, Few shot QA)    │
       └─────────────┬──────────────┘
                     ▼
       ┌────────────────────────────┐
       │   Text & Metadata Embedder │
       └─────────────┬──────────────┘
                     ▼
       ┌────────────────────────────┐
       │     Create vector index    │
       └─────────────┬──────────────┘
                     ▼
       ┌────────────────────────────┐
       │    Retrieval + Reranking   │
       └─────────────┬──────────────┘
                     ▼
       ┌────────────────────────────┐
       │     LLM response synthesis │
       └────────────────────────────┘



---

## ⚙️ Installation

**Prerequisites**  
- Python **3.11+**  
- [`uv`](https://github.com/astral-sh/uv) – Python package manager
- A **MongoDB Atlas** cluster (free tier works) – [create one here](https://www.mongodb.com/atlas)
- [Ollama](https://ollama.com) running locally for the LLM (`ollama serve`)

```bash
git clone https://github.com/ISHANT-GUPTA/doX-Q.git

cd doX-Q

uv sync --locked
```

**Configure credentials**

Copy the example env file and fill in your own values. The real `.env` is
git-ignored, so your credentials never get committed:

```bash
cp .env.example .env
# then edit .env and set MONGODB_URI (from Atlas > Connect > Drivers)
```

`.env` keys:

| Key | Purpose |
|-----|---------|
| `MONGODB_URI` | Atlas connection string (`mongodb+srv://…`) |
| `MONGODB_DB` | Database name (default `doxq_db`) |
| `SESSION_SECRET` | Random string used to sign web sessions |

**Initialize and run**

```bash
uv run python setup_db.py          # verifies the Atlas connection + creates indexes
uv run uvicorn main:app --reload   # then open http://localhost:8000
```

---
## 🚀 Roadmap

Planned enhancements to make **doX-Q** even more powerful and production-ready:

- **📊 Configurable Retrieval Strategies**  
  Hybrid search (**dense embeddings + BM25**) and metadata-filtered retrieval.

- **🛠 Modular Architecture**  
  Easily swap components like embedding models, databases, or LLM backends.

- **🧠 Adaptive Reranking**  
  Introduce machine-learning–driven reranking strategies that adapt based on user feedback.

- **🔗 Knowledge Graph Integration**  
  Enrich embeddings and retrieval with semantic relationships between entities for deeper reasoning capabilities.

