"""Persistent on-disk cache for fetched company data.

Streamlit's ``st.cache_data`` keeps results in memory only — they are lost on
every restart / Cloud redeploy, so each cold start re-hits the external APIs.
This module adds a small disk-backed layer underneath that in-memory cache:

    in-memory (st.cache_data, fast)  →  disk (this module, persistent)  →  API

Entries are pickled together with a write timestamp and considered fresh until
``ttl`` seconds elapse, after which they are transparently re-fetched.

Configuration (environment variables):
    HANSE_CACHE_DIR       directory for cache files            (default: ".cache")
    HANSE_CACHE_TTL       global TTL override in seconds; if >0 it replaces the
                          per-call ttl, letting you tune freshness without code
                          changes                              (default: unset)
    HANSE_CACHE_DISABLED  set to "1"/"true" to bypass the disk cache entirely
"""
from __future__ import annotations

import os
import time
import pickle
import shutil
import hashlib
import pathlib
from typing import Any, Callable

CACHE_DIR = pathlib.Path(os.environ.get("HANSE_CACHE_DIR", ".cache"))


def _disabled() -> bool:
    return os.environ.get("HANSE_CACHE_DISABLED", "").lower() in ("1", "true", "yes")


def _effective_ttl(ttl: float | None) -> float | None:
    """A positive HANSE_CACHE_TTL overrides the per-call ttl globally."""
    override = os.environ.get("HANSE_CACHE_TTL")
    if override:
        try:
            v = float(override)
            if v > 0:
                return v
        except ValueError:
            pass
    return ttl


def _path(namespace: str, key: str) -> pathlib.Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    safe_ns = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)
    return CACHE_DIR / safe_ns / f"{digest}.pkl"


def _is_empty(value: Any) -> bool:
    """Don't persist 'no data' results — they are usually transient (a throttled
    API returning an empty frame) and caching them would suppress retries."""
    if value is None:
        return True
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            return value.empty
    except Exception:
        pass
    if isinstance(value, (list, dict, str, tuple)) and len(value) == 0:
        return True
    return False


def load(namespace: str, key: str, ttl: float | None) -> tuple[Any, bool]:
    """Return (value, hit). ``hit`` is False on miss, expiry or read error."""
    if _disabled():
        return None, False
    path = _path(namespace, key)
    if not path.exists():
        return None, False
    try:
        with path.open("rb") as fh:
            ts, value = pickle.load(fh)
    except Exception:
        return None, False
    ttl = _effective_ttl(ttl)
    if ttl is not None and ttl >= 0 and (time.time() - ts) > ttl:
        return None, False
    return value, True


def store(namespace: str, key: str, value: Any) -> None:
    if _disabled() or _is_empty(value):
        return
    path = _path(namespace, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump((time.time(), value), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)  # atomic on POSIX
    except Exception:
        # Caching must never break the app — a failed write just means a miss.
        pass


def cached(namespace: str, key: str, ttl: float | None, producer: Callable[[], Any]) -> Any:
    """Return cached value when fresh, else call ``producer`` and persist it.

    Exceptions raised by ``producer`` propagate and nothing is stored, so
    transient API errors are never cached.
    """
    value, hit = load(namespace, key, ttl)
    if hit:
        return value
    value = producer()
    store(namespace, key, value)
    return value


def clear() -> int:
    """Delete the whole disk cache. Returns the number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    n = sum(1 for _ in CACHE_DIR.rglob("*.pkl"))
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    return n


def stats() -> dict:
    """Lightweight cache stats for display (file count + total size)."""
    if not CACHE_DIR.exists():
        return {"files": 0, "bytes": 0}
    files = list(CACHE_DIR.rglob("*.pkl"))
    return {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
