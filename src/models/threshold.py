import json
from pathlib import Path
import numpy as np


def load_metadata(metadata_path: str | Path) -> dict:
    """
    Loads the metadata file
    """
    metadata_path = Path(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_threshold(metadata: dict) -> float:
    """
    Gets threshold from metadata file
    """
    if "threshold" not in metadata:
        raise KeyError("metadata.json missing required key: 'threshold'")
    return float(metadata["threshold"])


def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert probabilities into binary predictions using a threshold.
    Returns int array (0/1).
    """
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)