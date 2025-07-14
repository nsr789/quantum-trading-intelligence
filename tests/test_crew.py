from src.agents.crew_rag import qa_with_crew

def test_crew_offline():
    # Works even with no keys
    out = qa_with_crew("What does the company do?", "AAPL")
    assert isinstance(out, str) and out
