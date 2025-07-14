# ── replace tests/test_cache.py
import time, random

from src.data.cache import cached


@cached
def _slow(x: int) -> int:
    time.sleep(0.03)  # ~30 ms
    return x * x


def test_cache_speed():
    unique_arg = random.randint(0, 999_999)
    t0 = time.perf_counter()
    a = _slow(unique_arg)
    t1 = time.perf_counter()

    b = _slow(unique_arg)  # cached
    t2 = time.perf_counter()

    uncached, cached = t1 - t0, t2 - t1
    assert a == b
    assert cached * 3 < uncached, f"cache ineffective: {uncached:.4f}s vs {cached:.4f}s"
