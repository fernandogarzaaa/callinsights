"""Structured logging: JSON lines (default) or human-readable text, with request ids."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# Attributes every LogRecord has; anything else passed via ``extra=`` is emitted as a field.
_RESERVED = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in vars(record).items() if k not in _RESERVED and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, suitable for Loki, CloudWatch, Datadog, etc."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{datetime.fromtimestamp(record.created, tz=UTC).strftime('%Y-%m-%dT%H:%M:%S')} "
            f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        )
        rid = request_id_var.get()
        fields = _extra_fields(record)
        if rid:
            fields = {"request_id": rid, **fields}
        if fields:
            base += " " + " ".join(f"{k}={v}" for k, v in fields.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install a single stdout handler on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    handler.set_name("callinsights")
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == "callinsights":
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    # uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    # Our request middleware logs access lines; silence the duplicate uvicorn access log.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # Per-request client logs from httpx would leak provider URLs at INFO; keep them quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
