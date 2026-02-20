"""Message readers for various data formats."""

import json
from pathlib import Path


def read_discord_package_messages(filepath: str | Path) -> list[str]:
    """Read messages from a Discord data package JSON (list of messages)."""
    filepath = Path(filepath)
    data = json.loads(filepath.read_text())
    return [str(msg["Contents"]) for msg in data]
