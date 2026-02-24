"""Background task runner.

Default: tasks execute in a bounded ThreadPoolExecutor (zero setup).
Optional: run ``python worker.py`` as a separate process for dedicated
background processing (uses a shared queue via filesystem).

In both modes the API is the same — call ``worker.submit(fn, *args)``.
"""

from __future__ import annotations

import atexit
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

logger = logging.getLogger(__name__)

# Bounded pool prevents unbounded thread growth under load.
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bg-worker")


def submit(fn: Callable, *args, **kwargs) -> None:
    """Submit a background task.

    Runs in a bounded thread pool.  Fire-and-forget — exceptions are
    logged, never raised to the caller.
    """
    def _safe():
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Background task error: {e}")

    _pool.submit(_safe)


def shutdown(wait: bool = True) -> None:
    """Shut down the worker pool gracefully.

    Called automatically at interpreter exit.  Pass ``wait=False`` to
    cancel pending tasks immediately.
    """
    _pool.shutdown(wait=wait, cancel_futures=not wait)
    logger.info("Background worker pool shut down")


# Ensure the pool drains on normal interpreter shutdown.
atexit.register(shutdown, wait=True)
