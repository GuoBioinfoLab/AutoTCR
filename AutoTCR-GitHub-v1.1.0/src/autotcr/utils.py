"""Output-path helper shared by the public API and CLI."""
from pathlib import Path
from typing import Union


def ensure_parent(path: Union[str, Path]) -> Path:
    """Create the parent directory and return an absolute output path."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
