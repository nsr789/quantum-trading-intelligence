from src.agents.crew_report import generate_report
from src.data.cache import cached

def test_report_cached():
    # first call caches
    txt1 = generate_report("AAPL")
    # second call pulled from cache → same output
    txt2 = generate_report("AAPL")
    assert txt1 == txt2 and txt1.startswith("-")
