"""Evidence logging: per-frame recognition + command records to CSV and JSONL."""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

_FIELDS = [
    "timestamp_iso",
    "epoch",
    "speaker_id",
    "recognized",
    "confidence",
    "num_faces",
    "error_norm",
    "status",
    "command",
]


class EvidenceLogger:
    """Writes a timestamped CSV + JSONL pair per session under logs/."""

    def __init__(self, log_dir: str | Path, stamp: str | None = None):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.dir / f"session_{stamp}.csv"
        self.jsonl_path = self.dir / f"session_{stamp}.jsonl"

        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=_FIELDS)
        self._writer.writeheader()
        self._jsonl_file = open(self.jsonl_path, "w", encoding="utf-8")
        self.count = 0

    def log(
        self,
        *,
        speaker_id: str,
        recognized: bool,
        confidence: float,
        num_faces: int,
        error_norm: float,
        status: str,
        command: str,
    ):
        now = time.time()
        row = {
            "timestamp_iso": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
            "epoch": round(now, 3),
            "speaker_id": speaker_id,
            "recognized": int(bool(recognized)),
            "confidence": round(float(confidence), 4),
            "num_faces": int(num_faces),
            "error_norm": round(float(error_norm), 4),
            "status": status,
            "command": command,
        }
        self._writer.writerow(row)
        self._jsonl_file.write(json.dumps(row) + "\n")
        self.count += 1
        # Flush periodically so a crash still leaves usable evidence.
        if self.count % 15 == 0:
            self.flush()

    def flush(self):
        self._csv_file.flush()
        self._jsonl_file.flush()

    def close(self):
        self.flush()
        self._csv_file.close()
        self._jsonl_file.close()
