"""
embedder.py — SBERT embedding pipeline for doXQ Healthcare RAG.

Generates vector embeddings using sentence-transformers and stores them
in MongoDB Atlas via database.py. Replaces the old ChromaDB-based storage.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from database import save_document, save_embeddings_batch, update_document_chunk_count
from embedding_model import embeddings

# Configuration Constants
UPLOAD_DIR = "uploaded_pdfs"


def generate_vector_embeddings(filename: str, user_id: int) -> dict:
    """
    Generate vector embeddings using SBERT and store them in MongoDB.

    Parameters:
        filename (str): PDF filename in `uploaded_pdfs/`.
        user_id (int): The authenticated user's ID (for linking the document).

    Returns:
        dict: Success details including doc_id and num_chunks.
    """
    # Resolve PDF path
    pdf_path = os.path.join(UPLOAD_DIR, filename)

    # Handle missing .pdf extension automatically
    if not os.path.exists(pdf_path) and not filename.endswith(".pdf"):
        pdf_path_with_ext = pdf_path + ".pdf"
        if os.path.exists(pdf_path_with_ext):
            pdf_path = pdf_path_with_ext

    if not (os.path.exists(pdf_path) and os.path.isfile(pdf_path) and pdf_path.endswith(".pdf")):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Found PDF file: {pdf_path}. Commencing extraction & embedding...")

    # 1. Save document metadata to MongoDB
    original_name = os.path.basename(pdf_path)
    doc_id = save_document(user_id, filename, original_name)
    print(f"Document registered in DB with ID: {doc_id}")

    # 2. Load and parse the PDF file
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 3. Split PDF text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documents)
    print(f"Split PDF into {len(docs)} document chunks.")

    # 4. Generate embeddings for each chunk
    print(" Generating SBERT embeddings for all chunks...")
    texts = [doc.page_content for doc in docs]
    vectors = embeddings.embed_documents(texts)

    # 5. Prepare batch data and store in MongoDB
    chunks_data = []
    for i, (doc, vec) in enumerate(zip(docs, vectors)):
        page_num = doc.metadata.get("page", None)
        chunks_data.append({
            "chunk_index": i,
            "chunk_text": doc.page_content,
            "embedding": vec,
            "page_number": page_num,
        })

    print(f"Storing {len(chunks_data)} embeddings in MongoDB...")
    save_embeddings_batch(doc_id, chunks_data)

    # 6. Update chunk count on the document record
    update_document_chunk_count(doc_id, len(docs))

    print("Successfully stored all embeddings in MongoDB!")

    return {
        "status": "success",
        "type": "pdf_ingestion",
        "filename": original_name,
        "doc_id": doc_id,
        "num_chunks": len(docs),
    }
