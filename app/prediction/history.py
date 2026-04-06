"""Lightweight local prediction-history storage."""

from __future__ import annotations

import json
from pathlib import Path

from app.prediction.errors import PredictionHistoryError
from app.prediction.schemas import PredictionHistoryEntry


class PredictionHistoryStore:
    """Persist and query recent prediction jobs via newline-delimited JSON."""

    def __init__(self, history_path: Path) -> None:
        self._history_path = history_path

    def append(self, entry: PredictionHistoryEntry) -> None:
        """Append one prediction-history record to disk."""

        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            with self._history_path.open("a", encoding="utf-8") as handle:
                handle.write(entry.model_dump_json())
                handle.write("\n")
        except Exception as exc:
            raise PredictionHistoryError(f"Could not write prediction history: {exc}") from exc

    def list_recent(self, limit: int = 20) -> list[PredictionHistoryEntry]:
        """Return recent prediction jobs ordered newest-first."""

        if limit <= 0:
            return []
        if not self._history_path.exists():
            return []

        entries: list[PredictionHistoryEntry] = []
        try:
            lines = self._history_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                cleaned = line.strip()
                if not cleaned:
                    continue
                entries.append(PredictionHistoryEntry.model_validate(json.loads(cleaned)))
        except Exception as exc:
            raise PredictionHistoryError(f"Could not read prediction history: {exc}") from exc

        entries.sort(key=lambda item: item.timestamp, reverse=True)
        return entries[:limit]
