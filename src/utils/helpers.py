# Shared Utilities
"""
Common utility functions used across the project.
"""

import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent
