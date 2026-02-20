from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .threshold import load_metadata, get_threshold, apply_threshold


def load_model(model_path: str | Path):
    """
    Loads the saved pickle model 
    """
    model_path = Path(model_path)
    return joblib.load(model_path)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """
    Returns probability of class 1 (fraud).
    Assumes sklearn-like estimator/pipeline with predict_proba.
    """
    proba = model.predict_proba(X)[:, 1]
    return np.asarray(proba, dtype=float)


def predict_with_metadata(
    X: pd.DataFrame,
    model_path: str | Path = "../models/baseline_pipeline.joblib",
    metadata_path: str | Path = "../models/metadata.json",
) -> dict:
    """
    Loads model + metadata, returns proba, pred, threshold, model_version (if present).
    """
    model = load_model(model_path)
    metadata = load_metadata(metadata_path)

    threshold = get_threshold(metadata)
    proba = predict_proba(model, X)
    pred = apply_threshold(proba, threshold)

    return {
        "proba": proba,
        "pred": pred,
        "threshold": threshold,
        "model_version": metadata.get("model_version"),
        "model_name": metadata.get("model_name"),
    }