def test_page_import():
    """Ensure Streamlit page script imports without side-effects."""
    import importlib

    importlib.import_module("app.pages.1_📈_Price_Dashboard")
