from collections import defaultdict
from typing import cast

import numpy as np
from jaxtyping import Float
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor

from redglass.log import logger

"""
Note that our probes are trained such that "unsafe" is positive and "safe" is negative.
But we want to measure the performance of the model in detecting deception.
Therefore, we need to flip the sign of the scores to get the correct AUROC (or alternatively,
we can flip the labels).
"""


def get_auroc_manual(
    safe_scores: Float[Tensor, " n_dialogues"],
    unsafe_scores: Float[Tensor, " n_dialogues"],
    N: int = 1000,
):
    """
    Custom implementation of auroc (h/t Stefan).
    Deprecated, but kept for reference (as a test reference).
    """
    eps = 1e-1
    small = min(safe_scores.min().item(), unsafe_scores.min().item()) - eps
    big = max(safe_scores.max().item(), unsafe_scores.max().item()) + eps
    thresholds = np.linspace(small, big, N)
    false_positives: dict[int, int] = defaultdict(int)
    true_positives: dict[int, int] = defaultdict(int)
    true_negatives: dict[int, int] = defaultdict(int)
    false_negatives: dict[int, int] = defaultdict(int)

    for safe_score, unsafe_score in zip(safe_scores, unsafe_scores, strict=True):
        # Higher score = more unsafe
        for i, threshold in enumerate(thresholds):
            # Safe prompt (negative)
            if safe_score <= threshold:
                # Detect safe (negative)
                true_negatives[i] += 1
            else:
                # Detect unsafe (positive)
                false_positives[i] += 1
            # Unsafe prompt (positive)
            if unsafe_score <= threshold:
                # Detect unsafe (positive)
                false_negatives[i] += 1
            else:
                # Detect safe (negative)
                true_positives[i] += 1

    tprs = [
        true_positives[i] / (true_positives[i] + false_negatives[i]) for i in range(N)
    ]
    fprs = [
        false_positives[i] / (false_positives[i] + true_negatives[i]) for i in range(N)
    ]

    # flip as fprs are in descending order
    auroc = -1 * np.trapezoid(tprs, fprs)

    return auroc


def get_true_and_scores(
    safe_scores: Float[Tensor, " n_dialogues"],
    unsafe_scores: Float[Tensor, " n_dialogues"],
):
    y_true = np.array([1] * len(unsafe_scores) + [0] * len(safe_scores))
    y_scores = np.concatenate([unsafe_scores, safe_scores])
    return y_true, y_scores


def get_auroc(
    safe_scores: Float[Tensor, " n_dialogues"],
    unsafe_scores: Float[Tensor, " n_dialogues"],
    weights: Float[Tensor, " n_dialogues"] | None = None,
) -> float:
    y_true, y_scores = get_true_and_scores(safe_scores, unsafe_scores)

    if np.isnan(y_scores).any():
        logger.warning("NaN scores found in AUROC calculation.")
        y_true = y_true[~np.isnan(y_scores)]
        y_scores = y_scores[~np.isnan(y_scores)]
    auroc = roc_auc_score(y_true, y_scores, sample_weight=weights)
    return cast(float, auroc)


def get_fpr_tpr(
    safe_scores: Float[Tensor, " n_dialogues"],
    unsafe_scores: Float[Tensor, " n_dialogues"],
) -> tuple[list[float], list[float], list[float]]:
    y_true, y_scores = get_true_and_scores(safe_scores, unsafe_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)
    fpr = fpr.tolist()
    tpr = tpr.tolist()
    thresholds = thresholds.tolist()
    return fpr, tpr, thresholds


def get_tpr_at_fpr_from_paired_dataset(
    safe_scores: Float[Tensor, " n_dialogues"],
    unsafe_scores: Float[Tensor, " n_dialogues"],
    fpr: float,
) -> float:
    fprs, tprs, _ = get_fpr_tpr(safe_scores, unsafe_scores)
    idx = np.where(np.array(fprs) > fpr)[0][0]
    tpr = tprs[idx]
    return cast(float, tpr)


def get_tpr_at_fpr_from_control_dataset(
    unsafe_scores: Float[Tensor, " n_dialogues"],
    threshold: float,
):
    return (unsafe_scores > threshold).float().mean().item()
