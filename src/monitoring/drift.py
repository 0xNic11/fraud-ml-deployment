import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """
    Population Stability Index using quantile bins from expected.
    Recommended: compute PSI on log-transformed skewed features like Amount.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Quantile-based bins from expected
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf

    e_counts = np.histogram(expected, bins=quantiles)[0]
    a_counts = np.histogram(actual, bins=quantiles)[0]

    e_perc = e_counts / max(e_counts.sum(), 1)
    a_perc = a_counts / max(a_counts.sum(), 1)

    e_perc = np.clip(e_perc, eps, 1.0)
    a_perc = np.clip(a_perc, eps, 1.0)

    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def psi_category(psi_value: float) -> str:
    """
    Common rule of thumb:
    < 0.1: no drift
    0.1–0.25: moderate drift
    > 0.25: significant drift
    """
    if psi_value < 0.1:
        return "low"
    if psi_value < 0.25:
        return "moderate"
    return "high"