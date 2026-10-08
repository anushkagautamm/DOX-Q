"""
setup_db.py — Initializes the MongoDB Atlas database for doXQ.
Run this script once to verify connectivity and create indexes before
running the app. (The app also ensures indexes lazily on first use, so
this is a convenience/sanity check, not a hard requirement.)
"""

from dotenv import load_dotenv
from database import get_db

load_dotenv()


def init_db():
    """Connect to Atlas and create all required indexes if they don't exist."""
    print("⏳ Initializing database...")
    db = get_db()  # connects and calls ensure_indexes()
    print(f"Connected to MongoDB database '{db.name}'.")
    print(f"Collections: {sorted(db.list_collection_names()) or '(created on first write)'}")
    print("✅ Database initialized successfully.")


if __name__ == "__main__":
    init_db()
