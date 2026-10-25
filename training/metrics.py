"""Official UPAR 2027 metrics for local validation (both tracks).

Track 1: mA, Label_F1, Instance_{Acc,Prec,Rec,F1}, Challenge_Avg = harm.mean(mA, Inst_F1)
Track 2: mADM (ranking metric), mAP, mINP, Rank-{1,5,10}
"""
from __future__ import annotations

import numpy as np  # noqa: F401 (re-exported for wrappers)


# ---------------------------------------------------------------- Track 1

def track1_metrics(gt: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """gt: [N,40] 0/1; probs: [N,40] in [0,1]; decision at p > 0.5."""
    preds = (probs > 0.5).astype(np.int64)
    gt = gt.astype(np.int64)
    TP = ((preds == 1) & (gt == 1)).sum(0).astype(np.float64)
    FN = ((preds == 0) & (gt == 1)).sum(0).astype(np.float64)
    TN = ((preds == 0) & (gt == 0)).sum(0).astype(np.float64)
    FP = ((preds == 1) & (gt == 0)).sum(0).astype(np.float64)
    mA = float(np.mean((TP / np.maximum(TP + FN, 1) + TN / np.maximum(TN + FP, 1)) / 2))
    prec = TP / np.maximum(TP + FP, 1)
    rec = TP / np.maximum(TP + FN, 1)
    label_f1 = float(np.mean(2 * prec * rec / np.maximum(prec + rec, 1e-12)))
    inter = (preds & gt).sum(1).astype(np.float64)
    p_sz = preds.sum(1).astype(np.float64)
    g_sz = gt.sum(1).astype(np.float64)
    i_prec = inter / np.maximum(p_sz, 1)
    i_rec = inter / np.maximum(g_sz, 1)
    i_f1 = 2 * i_prec * i_rec / np.maximum(i_prec + i_rec, 1e-12)
    i_acc = inter / np.maximum(p_sz + g_sz - inter, 1)
    return {
        "mA": mA, "Label_F1": label_f1,
        "Instance_Acc": float(i_acc.mean()), "Instance_Prec": float(i_prec.mean()),
        "Instance_Rec": float(i_rec.mean()), "Instance_F1": float(i_f1.mean()),
        "Challenge_Avg": float(2 * mA * i_f1.mean() / max(mA + i_f1.mean(), 1e-12)),
    }


def per_attribute_accuracy(gt: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Per-attribute (TPR+TNR)/2 - diagnostic for calibration."""
    gt = gt.astype(np.int64)
    preds = preds.astype(np.int64)
    TP = ((preds == 1) & (gt == 1)).sum(0)
    FN = ((preds == 0) & (gt == 1)).sum(0)
    TN = ((preds == 0) & (gt == 0)).sum(0)
    FP = ((preds == 1) & (gt == 0)).sum(0)
    return (TP / np.maximum(TP + FN, 1) + TN / np.maximum(TN + FP, 1)) / 2


# ---------------------------------------------------------------- Track 2

def _rank_metrics(D: np.ndarray, qids: np.ndarray, GT: np.ndarray,
                  max_queries: int | None = None) -> dict[str, float]:
    order = np.argsort(D, axis=1, kind="stable")
    nq_total = D.shape[0]
    idx = np.arange(nq_total)
    if max_queries is not None and nq_total > max_queries:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(nq_total, max_queries, replace=False))
    mADM = mAP = mINP = 0.0
    r1 = r5 = r10 = 0
    n_eval = 0
    for qi in idx:
        correct = (qids == qi)
        R = int(correct.sum())
        if R == 0:
            continue
        n_eval += 1
        ranking = order[qi]
        ranks = np.where(correct[ranking])[0] + 1  # 1-based
        k = np.arange(1, R + 1)
        mAP += (k / ranks).sum() / R
        mINP += R / ranks[-1]
        r1 += ranks[0] <= 1
        r5 += ranks[0] <= 5
        r10 += ranks[0] <= 10
        q = GT[qids == qi][0]
        dom = (q[None, :] == GT).mean(1)
        mean_dom, max_dom = dom.mean(), dom.max()
        ndom = np.maximum(0.0, (dom - mean_dom) / (max_dom - mean_dom + 1e-12))
        mADM += sum(ndom[ranking[:r]].sum() / r for r in ranks) / R
    return {
        "mADM": mADM / n_eval, "mAP": mAP / n_eval, "mINP": mINP / n_eval,
        "Rank-1": r1 / n_eval, "Rank-5": r5 / n_eval, "Rank-10": r10 / n_eval,
        "n_eval": float(n_eval),
    }


def track2_metrics(D: np.ndarray, qids: np.ndarray, GT: np.ndarray,
                   max_queries: int | None = None) -> dict[str, float]:
    """D: [Q,G] distances (smaller=better); qids: [G] query index per gallery image;
    GT: [G,40] gallery ground truth."""
    return _rank_metrics(D, qids, GT, max_queries)


def l1_distances(queries: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """sum|q - p| without the QxGx40 tensor:  sum(p) + q . (1 - 2p)"""
    probs = probs.astype(np.float32)
    return probs.sum(1)[None, :] + queries.astype(np.float32) @ (1.0 - 2.0 * probs).T


def adm_weighted_distances(queries: np.ndarray, probs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted L1 with per-attribute weights w (len 40)."""
    probs = probs.astype(np.float32)
    q = queries.astype(np.float32)
    # sum_j w_j |q_j - p_j| = sum_j w_j ( q_j(1-2p_j) + p_j )
    per_img = (probs * w).sum(1)                # [G]  constant per gallery image
    per_q = ((1.0 - 2.0 * probs) * w) @ q.T     # [G,Q] query-dependent part
    return (per_img[None, :] + per_q.T).astype(np.float32)
