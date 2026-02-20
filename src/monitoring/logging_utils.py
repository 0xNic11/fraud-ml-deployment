import json
from pathlib import Path
from datetime import datetime
import numpy as np


def _to_builtin(x):
    # Convert numpy types to Python types for JSON
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def log_predictions(
    log_path: str | Path,
    model_name: str | None,
    model_version: str | None,
    threshold: float,
    proba: np.ndarray,
    pred: np.ndarray,
    extra: dict | None = None,
) -> None:
    """
    Appends JSONL (one JSON object per line). Good for monitoring pipelines.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": model_name,
        "model_version": model_version,
        "threshold": float(threshold),
        "n_scored": int(len(proba)),
        "proba_summary": {
            "min": float(np.min(proba)),
            "p50": float(np.median(proba)),
            "p95": float(np.quantile(proba, 0.95)),
            "max": float(np.max(proba)),
            "mean": float(np.mean(proba)),
        },
        "pred_summary": {
            "positive_rate": float(np.mean(pred)),
            "n_positive": int(np.sum(pred)),
        },
    }

    if extra:
        record["extra"] = {k: _to_builtin(v) for k, v in extra.items()}

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
