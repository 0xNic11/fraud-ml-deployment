import json
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_model(model, X_test, y_test, save_path: str):
    """Model Evaluation Function

    Args:
        model (pkl): Model used for evaluation
        X_test (DataFrame): X_test set
        y_test (Dataframe): y_test set
        save_path (str): Metrics path

    Returns:
        Dictionary: Model Metrics
    """

    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba)
    }

    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    return metrics