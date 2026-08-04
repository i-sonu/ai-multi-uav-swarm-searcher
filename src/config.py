"""Configuration loading.

All constants live in ``configs/*.yaml`` (CLAUDE.md §3: no hardcoded paths or
magic numbers in source). This module loads a YAML config into a plain dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels up from this file (src/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file into a dict.

    Args:
        path: config file to load; defaults to ``configs/default.yaml``.

    Returns:
        Parsed configuration as a nested dict.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
