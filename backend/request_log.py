"""In-memory log of failed API requests, shown on the Activity page.

Every response from an ``/api/`` route with status 400 or higher is recorded
(method, path, status, the JSON error message and the duration). Request
bodies and client addresses are not stored. The log is bounded and lives in
the single backend process, so it clears on restart like the step history.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from flask import Flask, Response, current_app, g, request

MAX_ENTRIES = 200
MAX_MESSAGE_CHARS = 200
# The log endpoint itself is not recorded, so a bad query to it cannot flood the log.
EXCLUDED_PATHS = {"/api/v1/system/errors"}


class RequestErrorLog:
    """Thread-safe ring buffer of failed request records."""

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self._entries: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._total = 0

    def record(self, method: str, path: str, status: int, error: str | None, duration_ms: float | None) -> None:
        entry = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "path": path,
            "status": int(status),
            "error": (error or "")[:MAX_MESSAGE_CHARS] or None,
            "duration_ms": None if duration_ms is None else round(duration_ms, 2),
        }
        with self._lock:
            self._entries.append(entry)
            self._total += 1

    def recent(self, limit: int) -> list[dict[str, Any]]:
        """Newest first."""
        with self._lock:
            items = list(self._entries)
        return list(reversed(items))[:limit]

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total = 0


def get_request_log() -> RequestErrorLog:
    log = current_app.config.get("REQUEST_ERROR_LOG")
    if log is None:
        log = RequestErrorLog()
        current_app.config["REQUEST_ERROR_LOG"] = log
    return log


def _error_message(response: Response) -> str | None:
    if not response.is_json:
        return None
    try:
        body = response.get_json(silent=True)
    except Exception:
        return None
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        return body["error"]
    return None


def init_request_log(app: Flask) -> None:
    """Attach the hooks that record failed /api/ requests."""
    app.config.setdefault("REQUEST_ERROR_LOG", RequestErrorLog())

    @app.before_request
    def _start_timer() -> None:
        g._request_started = time.perf_counter()

    @app.after_request
    def _record_failure(response: Response) -> Response:
        path = request.path
        if response.status_code >= 400 and path.startswith("/api/") and path not in EXCLUDED_PATHS:
            started = g.get("_request_started")
            duration = None if started is None else (time.perf_counter() - started) * 1000.0
            try:
                get_request_log().record(request.method, path, response.status_code, _error_message(response), duration)
            except Exception:
                # Logging must never turn an error response into a different one.
                pass
        return response
