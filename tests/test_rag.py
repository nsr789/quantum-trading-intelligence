from src.knowledge.vectorstore import load_vectorstore

def test_vectorstore_seed():
    col = load_vectorstore()
    assert col.count() >= 3
