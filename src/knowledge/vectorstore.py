# ── src/knowledge/vectorstore.py ─────────────────────────────────────────────
"""
Light-weight Chroma wrapper.

Changes v2
* Added `_RICH_SNIPPETS` with extra facts (latest products, CEOs, etc.)
* Idempotent client helper `_client()` —
  creates a writable PersistentClient when possible, else in-memory.
* On first load, automatically (re)seeds the `company_docs` collection
  if the schema is empty or incompatible.
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── paths & embedding --------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
STORE_PATH = _ROOT / ".cache" / "chroma"
_EMBED = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# ── richer static corpus -----------------------------------------------------
_SEED = [
    # Apple
    ("Apple Inc. designs, manufactures and markets smartphones.", "AAPL"),
    ("As of July 2025 the newest iPhone is the **iPhone 16 Pro Max** (A18-based).", "AAPL"),
    # Microsoft
    ("Microsoft develops Windows, Office 365 and Azure cloud services.", "MSFT"),
    ("The current CEO of Microsoft is Satya Nadella.", "MSFT"),
    # Alphabet
    ("Alphabet is the parent of Google, focusing on internet services.", "GOOGL"),
    ("Google’s flagship smartphone is the Pixel 9 series (Tensor G4).", "GOOGL"),
]

def _default_docs() -> List[dict]:
    return [
        {"content": txt, "ticker": tic, "id": f"doc-{i}"}
        for i, (txt, tic) in enumerate(_SEED)
    ]

# ── client helper ------------------------------------------------------------
def _client() -> chromadb.api.ClientAPI:
    """Writable persistent client when FS is RW; else in-memory fallback."""
    try:
        return chromadb.PersistentClient(
            path=str(STORE_PATH), settings=Settings(allow_reset=True)
        )
    except RuntimeError:
        # Streamlit Community Cloud → read-only filesystem
        return chromadb.Client(Settings())

# ── public loader ------------------------------------------------------------
def load_vectorstore():
    """Return a collection seeded with rich snippets (autocreates if needed)."""
    client = _client()

    # If an old incompatible collection exists, wipe & recreate
    if "company_docs" in [c.name for c in client.list_collections()]:
        coll = client.get_collection("company_docs")
        try:
            # This will raise if EF config mismatches => delete & rebuild
            coll.count()
            return coll
        except Exception:  # noqa: BLE001
            client.delete_collection("company_docs")

    # Fresh collection
    coll = client.get_or_create_collection(
        "company_docs", embedding_function=_EMBED
    )
    docs = _default_docs()
    coll.add(
        documents=[d["content"] for d in docs],
        metadatas=[{"ticker": d["ticker"]} for d in docs],
        ids=[d["id"] for d in docs],
    )
    return coll
