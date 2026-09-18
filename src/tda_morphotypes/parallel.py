"""Minimal process-based parallel map with progress reporting."""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

_shared = None


def _set_shared(value) -> None:
    global _shared
    _shared = value


def shared():
    """Object passed as ``shared`` to :func:`parallel_map`, available in the workers."""
    return _shared


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 1))


def _report(desc: str, done: int, total: int, start: float) -> None:
    if not desc:
        return
    elapsed = time.time() - start
    eta = elapsed / done * (total - done) if done else 0.0
    end = "\n" if done == total else "\r"
    print(f"  {desc}: {done}/{total}  elapsed {elapsed:6.0f}s  remaining ~{eta:6.0f}s", end=end,
          file=sys.stderr, flush=True)


def parallel_map(func, items, n_jobs: int = 1, desc: str = "", shared=None, chunksize: int = 1) -> list:
    """``[func(item) for item in items]`` computed by ``n_jobs`` processes.

    ``shared`` is sent once to every worker and read there with :func:`shared`.
    """
    items = list(items)
    total, start = len(items), time.time()
    results = []
    step = max(1, total // 200)
    if n_jobs <= 1:
        _set_shared(shared)
        for k, item in enumerate(items, 1):
            results.append(func(item))
            if k % step == 0 or k == total:
                _report(desc, k, total, start)
        return results
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_set_shared, initargs=(shared,)) as pool:
        for k, result in enumerate(pool.map(func, items, chunksize=chunksize), 1):
            results.append(result)
            if k % step == 0 or k == total:
                _report(desc, k, total, start)
    return results
