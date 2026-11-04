"""Registro de auditoría local en JSONL: un evento por nodo del grafo.

Funciona sin red ni Langfuse. No guarda textos del CV ni de la oferta, solo decisiones,
métricas y conteos; el candidato se identifica con un hash de su ID (que deriva del nombre).
"""

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AUDIT_FILE = "audit.jsonl"
MAX_BYTES = 5 * 1024 * 1024  # Al superar este tamaño se rota a audit.jsonl.1


def anonymize(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


class AuditLog:
    def __init__(self, log_dir: Path | None, run_id: str, thread_id: str | None = None):
        # log_dir=None desactiva la escritura (ej. en tests que no la necesitan)
        self.path = Path(log_dir) / AUDIT_FILE if log_dir else None
        self.run_id = run_id
        self.thread_id = thread_id
        self.events: list[dict[str, Any]] = []
        # Los nodos paralelos del grafo pueden registrar al mismo tiempo
        self._lock = threading.Lock()

    def record(self, node: str, **details: Any) -> dict[str, Any]:
        event = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "node": node,
            **details,
        }
        with self._lock:
            self.events.append(event)
            if self.path:
                self._write(event)
        return event

    def _write(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > MAX_BYTES:
            self.path.replace(self.path.with_name(AUDIT_FILE + ".1"))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
