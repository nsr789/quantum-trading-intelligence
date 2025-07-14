from pathlib import Path

from src.config.settings import settings


def test_env_loading():
    readme = Path(settings.CACHE_DIR).parent / "README.md"
    assert readme.exists(), f"Expected README at {readme}"
