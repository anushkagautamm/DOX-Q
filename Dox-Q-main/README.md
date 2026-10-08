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



