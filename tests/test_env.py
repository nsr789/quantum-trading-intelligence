"""
Quick import test to verify Phase 1 environment.
Run with:  python -m tests.test_env
"""
core_packages = [
    "pandas", "numpy", "yfinance", "aiohttp", "openai", "groq",
    "langchain", "crewai", "chromadb", "sentence_transformers",
    "torch", "sklearn", "xgboost", "onnxruntime", "streamlit",
    "plotly", "requests", "praw"
]

def main():
    failed = []
    for pkg in core_packages:
        try:
            __import__(pkg)
        except Exception as e:
            failed.append((pkg, str(e)))
    if failed:
        print("❌  Missing packages:")
        for pkg, err in failed:
            print(f"   {pkg:<25} -> {err}")
    else:
        print("✅  All core packages import correctly.")

if __name__ == "__main__":
    main()
