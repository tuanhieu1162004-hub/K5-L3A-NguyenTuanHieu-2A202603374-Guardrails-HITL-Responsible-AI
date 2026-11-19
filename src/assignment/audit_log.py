"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open = defaultdict(deque)

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Record a redacted input and start timing the request."""
        self._open[user_id, request_id].append(
            (utc_now_iso(), monotonic(), content_filter(text)["redacted"])
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Record the response and correlate it with its input."""
        key = (user_id, request_id)
        pending = self._open.get(key)
        timestamp, started, user_input = pending.popleft() if pending else (None, None, None)
        if not pending:
            self._open.pop(key, None)
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": timestamp or utc_now_iso(),
            "input": user_input,
            "output": content_filter(text)["redacted"],
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round((monotonic() - started) * 1000, 3) if started is not None else None,
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
