"""Load and access the project configuration (config.json)."""
from __future__ import annotations

import json
import os
from pathlib import Path

# Project root = parent of this src/ directory.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.json"


class Config:
    """Thin dict wrapper with dotted-path access and absolute path helpers."""

    def __init__(self, data: dict, path: Path):
        self._data = data
        self.path = path

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(p, "r", encoding="utf-8") as f:
            return cls(json.load(f), p)

    def get(self, dotted: str, default=None):
        node = self._data
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def __getitem__(self, key):
        return self._data[key]

    def abspath(self, relative: str) -> Path:
        """Resolve a config-relative path against the project root."""
        p = Path(relative)
        return p if p.is_absolute() else (ROOT / p)
