"""
CAFA evaluation metrics.

Implements protein-centric metrics following the official CAFA protocol:
  - Fmax (maximum F-measure over precision-recall curve)
  - AUPR (area under precision-recall curve)
  - Smin (minimum semantic distance)
  - Coverage (fraction of proteins with at least one predicted term)

References:
  Jiang et al. (2016) An expanded evaluation of protein function prediction methods.
  https://doi.org/10.1186/s13059-016-1037-6
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


# ---------------------------------------------------------------------------
# Core CAFA metrics
# ---------------------------------------------------------------------------

def compute_fmax(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_thresholds: int = 101,
) -> tuple[float, float]:
    """
    Compute protein-centric Fmax.

    For each threshold t in [0, 1]:
        For each protein: precision(t), recall(t) are computed.
        Average precision and recall across proteins with at least one
        predicted term (for precision) or at least one true term (for recall).
        F(t) = 2 * P(t) * R(t) / (P(t) + R(t))
    Fmax = max over t of F(t).

    Args:
        y_true: binary (n_proteins, n_terms)
        y_score: float scores (n_proteins, n_terms)
        num_thresholds: number of threshold steps

    Returns:
        (fmax, best_threshold)
    """
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    fmax = 0.0
    best_t = 0.0

    for t in thresholds:
        y_pred = (y_score >= t).astype(float)

        # precision: over proteins that predicted ≥1 term
        has_pred = y_pred.sum(axis=1) > 0
        if has_pred.sum() > 0:
            tp = (y_true[has_pred] * y_pred[has_pred]).sum(axis=1)
            n_pred = y_pred[has_pred].sum(axis=1).clip(min=1)
            prec = (tp / n_pred).mean()
        else:
            prec = 0.0

        # recall: over proteins that have ≥1 true term
        has_true = y_true.sum(axis=1) > 0
        if has_true.sum() > 0:
            tp_r = (y_true[has_true] * y_pred[has_true]).sum(axis=1)
            n_true = y_true[has_true].sum(axis=1).clip(min=1)
            rec = (tp_r / n_true).mean()
        else:
            rec = 0.0

        if prec + rec > 0:
            f = 2 * prec * rec / (prec + rec)
            if f > fmax:
                fmax = f
                best_t = t

    return float(fmax), float(best_t)


def compute_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Micro-averaged area under the precision-recall curve.
    Flattens all labels and scores for a single AUPR value.
    """
    y_true_flat = y_true.ravel()
    y_score_flat = y_score.ravel()
    if y_true_flat.sum() == 0:
        return 0.0
    return float(average_precision_score(y_true_flat, y_score_flat))


def compute_coverage(y_score: np.ndarray, threshold: float) -> float:
    """Fraction of proteins with at least one predicted GO term above threshold."""
    has_pred = (y_score >= threshold).any(axis=1)
    return float(has_pred.mean())


def compute_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_thresholds: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute protein-centric precision and recall curves.

    Returns (precisions, recalls, thresholds) arrays.
    """
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    precisions = np.zeros(num_thresholds)
    recalls = np.zeros(num_thresholds)

    for i, t in enumerate(thresholds):
        y_pred = (y_score >= t).astype(float)
        has_pred = y_pred.sum(axis=1) > 0
        has_true = y_true.sum(axis=1) > 0

        if has_pred.sum() > 0:
            tp = (y_true[has_pred] * y_pred[has_pred]).sum(axis=1)
            n_pred = y_pred[has_pred].sum(axis=1).clip(min=1)
            precisions[i] = (tp / n_pred).mean()

        if has_true.sum() > 0:
            tp_r = (y_true[has_true] * y_pred[has_true]).sum(axis=1)
            n_true = y_true[has_true].sum(axis=1).clip(min=1)
            recalls[i] = (tp_r / n_true).mean()

    return precisions, recalls, thresholds


def compute_smin(
    y_true: np.ndarray,
    y_score: np.ndarray,
    term_ic: np.ndarray,
    num_thresholds: int = 101,
) -> tuple[float, float]:
    """
    Compute Smin (minimum semantic distance) using information content.

    S(t) = sqrt( ru(t)^2 + mi(t)^2 )
    where ru = remaining uncertainty, mi = misinformation.

    Args:
        y_true: binary (n_proteins, n_terms)
        y_score: float scores (n_proteins, n_terms)
        term_ic: information content per GO term (n_terms,)
        num_thresholds: threshold steps

    Returns:
        (smin, best_threshold)
    """
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    smin = np.inf
    best_t = 0.0

    for t in thresholds:
        y_pred = (y_score >= t).astype(float)
        fn = y_true * (1 - y_pred)  # false negatives
        fp = (1 - y_true) * y_pred  # false positives

        ru = (fn * term_ic).sum(axis=1).mean()   # remaining uncertainty
        mi = (fp * term_ic).sum(axis=1).mean()   # misinformation
        s = np.sqrt(ru ** 2 + mi ** 2)

        if s < smin:
            smin = s
            best_t = t

    return float(smin), float(best_t)


# ---------------------------------------------------------------------------
# Per-ontology evaluation
# ---------------------------------------------------------------------------

def evaluate_ontology(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ontology_name: str,
    term_ic: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """
    Compute all CAFA metrics for one ontology.

    Returns a metrics dict.
    """
    fmax, best_t = compute_fmax(y_true, y_score)
    aupr = compute_aupr(y_true, y_score)
    t = threshold if threshold is not None else best_t
    coverage = compute_coverage(y_score, t)

    metrics = {
        f"{ontology_name}/fmax": fmax,
        f"{ontology_name}/best_threshold": best_t,
        f"{ontology_name}/aupr": aupr,
        f"{ontology_name}/coverage": coverage,
    }

    if term_ic is not None:
        smin, _ = compute_smin(y_true, y_score, term_ic)
        metrics[f"{ontology_name}/smin"] = smin

    return metrics


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_thresholds: int = 101,
) -> float:
    """Return the threshold that maximises Fmax on the given set."""
    _, best_t = compute_fmax(y_true, y_score, num_thresholds=num_thresholds)
    return best_t
