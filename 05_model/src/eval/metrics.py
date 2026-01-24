"""Weighted precision/recall/Fmax utilities (simplified).

This mirrors Jiang et al. (2016) style with information accretion weights.
You will likely adjust for efficiency; this is a readable baseline.
"""
from __future__ import annotations
from typing import Dict, Iterable, Set, Tuple
import numpy as np

# Types: mapping protein -> set/iterable of term_ids
Truth = Dict[str, Set[str]]
Pred = Dict[str, Dict[str, float]]  # protein -> {term: score}
Weights = Dict[str, float]  # term -> ia


def weighted_pr(truth: Truth, pred: Pred, weights: Weights, threshold: float) -> Tuple[float, float]:
    tp = 0.0
    fp = 0.0
    fn = 0.0
    for protein, true_terms in truth.items():
        pred_terms = {t for t, s in pred.get(protein, {}).items() if s >= threshold}
        for t in pred_terms:
            tp += weights.get(t, 0.0) if t in true_terms else 0.0
            fp += weights.get(t, 0.0) if t not in true_terms else 0.0
        for t in true_terms:
            if t not in pred_terms:
                fn += weights.get(t, 0.0)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return precision, recall


def f_max(truth: Truth, pred: Pred, weights: Weights, thresholds: Iterable[float] | None = None) -> Tuple[float, float, float]:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 51)
    best_f = 0.0
    best_p = 0.0
    best_r = 0.0
    for th in thresholds:
        p, r = weighted_pr(truth, pred, weights, th)
        f = 2 * p * r / (p + r + 1e-12)
        if f > best_f:
            best_f, best_p, best_r = f, p, r
    return best_f, best_p, best_r
