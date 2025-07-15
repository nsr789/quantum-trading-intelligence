# ── tests/test_vectorstore.py ────────────────────────────────────────────────
from src.knowledge.vectorstore import load_vectorstore

def test_vectorstore_enriched():
    col = load_vectorstore()
    assert col.count() >= 6  # richer corpus seeded

    q = "latest iPhone"
    res = col.query(query_texts=[q], n_results=1)
    snippet = res["documents"][0][0].lower()
    assert "iphone" in snippet and "16" in snippet
