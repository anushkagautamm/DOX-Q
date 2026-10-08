"""
database.py — MongoDB Atlas database layer for doXQ Healthcare RAG.

Manages users, documents, vector embeddings (as binary float32 blobs), and
chat history. Uses pymongo with a single shared client (thread-safe, pooled).

Documents keep the integer `id` fields the MySQL schema used (allocated from
a `counters` collection), so sessions, API routes, and the frontend are
unchanged. Connection is configured via MONGODB_URI (and optionally
MONGODB_DB, default "doxq_db") in .env.
"""

import os
import struct
from datetime import datetime, timezone

import numpy as np
import bcrypt
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

load_dotenv()


# ── Connection Helper ────────────────────────────────────────────────────────

_client: MongoClient | None = None
_db = None


def get_db():
    """Return the shared database handle, connecting on first use."""
    global _client, _db
    if _db is None:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise RuntimeError(
                "MONGODB_URI is not set. Add your Atlas connection string to .env "
                "(mongodb+srv://<user>:<password>@<cluster>.mongodb.net/)."
            )
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000, appname="doXQ")
        _db = _client[os.getenv("MONGODB_DB", "doxq_db")]
        ensure_indexes(_db)
    return _db


def ensure_indexes(db) -> None:
    """Idempotently create the indexes the query paths rely on."""
    db.users.create_index([("username", ASCENDING)], unique=True, name="username")
    db.users.create_index([("email", ASCENDING)], unique=True, name="email")
    db.users.create_index([("id", ASCENDING)], unique=True)
    db.documents.create_index([("id", ASCENDING)], unique=True)
    db.documents.create_index([("user_id", ASCENDING), ("upload_date", DESCENDING)])
    db.embeddings.create_index([("document_id", ASCENDING)])
    db.chat_history.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])


def _next_id(sequence: str) -> int:
    """Allocate the next integer ID for a collection (MySQL AUTO_INCREMENT equivalent)."""
    counter = get_db().counters.find_one_and_update(
        {"_id": sequence},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return counter["seq"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── User Management ─────────────────────────────────────────────────────────

def create_user(username: str, email: str, password: str) -> int:
    """
    Register a new user with a bcrypt-hashed password.
    Returns the new user's ID.
    Raises ValueError if username or email already exists.
    """
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    user_id = _next_id("users")
    try:
        get_db().users.insert_one(
            {
                "id": user_id,
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "created_at": _utcnow(),
            }
        )
        return user_id
    except DuplicateKeyError as e:
        if "username" in str(e).lower():
            raise ValueError("Username already exists")
        elif "email" in str(e).lower():
            raise ValueError("Email already exists")
        else:
            raise ValueError("User already exists")


def authenticate_user(username: str, password: str) -> dict | None:
    """Verify username + password. Returns user dict on success, None on failure."""
    user = get_db().users.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return {"id": user["id"], "username": user["username"], "email": user["email"]}
    return None


def get_user_by_id(user_id: int) -> dict | None:
    """Fetch user info by ID."""
    return get_db().users.find_one(
        {"id": user_id},
        {"_id": False, "id": True, "username": True, "email": True, "created_at": True},
    )


# ── Document Management ─────────────────────────────────────────────────────

def save_document(user_id: int, filename: str, original_name: str) -> int:
    """Save document metadata and return the document ID."""
    doc_id = _next_id("documents")
    get_db().documents.insert_one(
        {
            "id": doc_id,
            "user_id": user_id,
            "filename": filename,
            "original_name": original_name,
            "upload_date": _utcnow(),
            "chunk_count": 0,
        }
    )
    return doc_id


def update_document_chunk_count(doc_id: int, chunk_count: int):
    """Update the chunk count for a document after embedding."""
    get_db().documents.update_one({"id": doc_id}, {"$set": {"chunk_count": chunk_count}})


def get_user_documents(user_id: int) -> list[dict]:
    """Get all documents for a user."""
    cursor = (
        get_db()
        .documents.find(
            {"user_id": user_id},
            {
                "_id": False,
                "id": True,
                "filename": True,
                "original_name": True,
                "upload_date": True,
                "chunk_count": True,
            },
        )
        .sort("upload_date", DESCENDING)
    )
    return list(cursor)


def delete_document(doc_id: int, user_id: int) -> bool:
    """Delete a document and its embeddings (explicit cascade). Returns True if deleted."""
    result = get_db().documents.delete_one({"id": doc_id, "user_id": user_id})
    if result.deleted_count == 0:
        return False
    get_db().embeddings.delete_many({"document_id": doc_id})
    return True


# ── Embedding Storage & Retrieval ───────────────────────────────────────────

def _serialize_vector(vec: list[float]) -> bytes:
    """Serialize a float vector to bytes for binary storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_vector(blob: bytes) -> np.ndarray:
    """Deserialize bytes back to a numpy float array."""
    n = len(blob) // 4  # 4 bytes per float32
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def save_embedding(doc_id: int, chunk_index: int, chunk_text: str, embedding_vector: list[float], page_number: int = None):
    """Store a single chunk embedding in MongoDB."""
    get_db().embeddings.insert_one(
        {
            "document_id": doc_id,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "embedding": _serialize_vector(embedding_vector),
            "page_number": page_number,
        }
    )


def save_embeddings_batch(doc_id: int, chunks: list[dict]):
    """
    Batch-insert embeddings for a document.
    Each chunk dict: {"chunk_index": int, "chunk_text": str, "embedding": list[float], "page_number": int|None}
    """
    if not chunks:
        return
    get_db().embeddings.insert_many(
        [
            {
                "document_id": doc_id,
                "chunk_index": c["chunk_index"],
                "chunk_text": c["chunk_text"],
                "embedding": _serialize_vector(c["embedding"]),
                "page_number": c.get("page_number"),
            }
            for c in chunks
        ]
    )


def get_embeddings_for_documents(doc_ids: list[int]) -> list[dict]:
    """
    Fetch all embeddings for a list of document IDs.
    Returns list of {"chunk_text": str, "embedding": np.ndarray, "page_number": int, "document_id": int}
    """
    if not doc_ids:
        return []

    cursor = get_db().embeddings.find(
        {"document_id": {"$in": doc_ids}},
        {"_id": False, "document_id": True, "chunk_text": True, "embedding": True, "page_number": True},
    )
    return [
        {
            "document_id": row["document_id"],
            "chunk_text": row["chunk_text"],
            "embedding": _deserialize_vector(row["embedding"]),
            "page_number": row["page_number"],
        }
        for row in cursor
    ]


# ── Chat History ────────────────────────────────────────────────────────────

def save_chat(user_id: int, message: str, reply: str, document_ids: list[int] = None):
    """Save a chat exchange."""
    get_db().chat_history.insert_one(
        {
            "user_id": user_id,
            "message": message,
            "reply": reply,
            "document_ids": document_ids if document_ids else None,
            "created_at": _utcnow(),
        }
    )


def get_chat_history(user_id: int, limit: int = 50) -> list[dict]:
    """Get recent chat history for a user."""
    cursor = (
        get_db()
        .chat_history.find(
            {"user_id": user_id},
            {"_id": False, "message": True, "reply": True, "document_ids": True, "created_at": True},
        )
        .sort("created_at", DESCENDING)
        .limit(limit)
    )
    rows = list(cursor)
    rows.reverse()  # chronological order
    return rows
