# Pattern: File System (강의05)
import json
from pathlib import Path

_RESULTS_DIR = Path("results")


def save_json(filename: str, data: dict) -> str:
    """Persist data as JSON under results/. Returns path string."""
    _RESULTS_DIR.mkdir(exist_ok=True)
    path = _RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(path)


def load_json(filename: str) -> dict:
    """Load JSON from results/. Raises FileNotFoundError if missing."""
    path = _RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"results/{filename} not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
