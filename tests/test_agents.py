from src.agents.tools import company_news_sentiment
from src.agents.crew import generate_report

def test_sentiment_shape():
    sent = company_news_sentiment("AAPL", limit=5)
    assert sum(sent.values()) == 1.0

def test_ai_report_runs():
    doc = generate_report("AAPL")
    assert isinstance(doc, str) and doc.strip()
