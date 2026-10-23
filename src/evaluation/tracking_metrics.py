"""HOTA, IDF1 and MOTA for frame-aligned object tracks.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import box_iou


def _arr(x, dtype):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _assignment(scores):
    if scores.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-scores)
        return list(zip(rows.tolist(), cols.tolist()))
    except ImportError as exc:
        raise ImportError("tracking metrics require scipy (scipy.optimize.linear_sum_assignment)") from exc


def _frame_matches(gt, pr, threshold):
    gb, gl, gi = (_arr(gt[k], d) for k, d in (("boxes", np.float32), ("labels", np.int64), ("track_ids", np.int64)))
    pb, pl, pi = (_arr(pr[k], d) for k, d in (("boxes", np.float32), ("labels", np.int64), ("track_ids", np.int64)))
    if len(gb) != len(gl) or len(gb) != len(gi) or len(pb) != len(pl) or len(pb) != len(pi):
        raise ValueError("boxes, labels and track_ids must have matching lengths")
    if not len(gb) or not len(pb):
        return [], len(gb), len(pb)
    ious = box_iou(torch.as_tensor(gb), torch.as_tensor(pb)).numpy()
    valid = (gl[:, None] == pl[None, :]) & (ious >= threshold)
    # Invalid pairs get zero; the final filtering enforces the IoU threshold.
    pairs = [(i, j) for i, j in _assignment(np.where(valid, ious, 0.0)) if valid[i, j]]
    return [((int(gl[i]), int(gi[i])), (int(pl[j]), int(pi[j])), float(ious[i, j]))
            for i, j in pairs], len(gb), len(pb)


def evaluate_tracking(gt_frames, pred_frames, iou_threshold=0.5):
    """Compute sequence-level HOTA, IDF1 and MOTA over aligned frames."""
    if len(gt_frames) != len(pred_frames):
        raise ValueError("ground-truth and prediction sequences must have the same frame count")
    g_counts, p_counts, pair_counts = {}, {}, {}
    total_gt = total_pred = id_switches = 0
    previous_assignment = {}
    frame_cache = []
    
    for frame_idx, (gt, pred) in enumerate(zip(gt_frames, pred_frames)):
        matches, ng, npred = _frame_matches(gt, pred, iou_threshold)
        total_gt += ng; total_pred += npred
        gb_labels = _arr(gt["labels"], np.int64)
        pb_labels = _arr(pred["labels"], np.int64)
        gids = _arr(gt["track_ids"], np.int64)
        pids = _arr(pred["track_ids"], np.int64)
        
        # Identities include class so a class change is not treated as same object.
        for cls, tid in zip(gb_labels, gids):
            key = (int(cls), int(tid)); g_counts[key] = g_counts.get(key, 0) + 1
        for cls, tid in zip(pb_labels, pids):
            key = (int(cls), int(tid)); p_counts[key] = p_counts.get(key, 0) + 1
        for gkey, pkey, iou in matches:
            pair_counts[(gkey, pkey)] = pair_counts.get((gkey, pkey), 0) + 1
            if gkey in previous_assignment and previous_assignment[gkey] != pkey:
                id_switches += 1
            previous_assignment[gkey] = pkey
        frame_cache.append(len(matches))

    # IDF1 uses a global identity assignment maximizing matched detections.
    gkeys, pkeys = list(g_counts), list(p_counts)
    contingency = np.zeros((len(gkeys), len(pkeys)), dtype=np.float64)
    for (g, p), count in pair_counts.items():
        contingency[gkeys.index(g), pkeys.index(p)] = count
    idtp = sum(contingency[i, j] for i, j in _assignment(contingency)) if contingency.size else 0.0
    idfn, idfp = total_gt - idtp, total_pred - idtp
    idf1 = 2 * idtp / (2 * idtp + idfn + idfp) if 2 * idtp + idfn + idfp else 0.0

    # HOTA: detection accuracy times association accuracy at each alpha.
    hota_values = []
    for alpha in np.arange(0.05, 1.0, 0.05):
        tp = fp = fn = 0
        alpha_pairs = {}
        for gt, pred in zip(gt_frames, pred_frames):
            matches, ng, npred = _frame_matches(gt, pred, alpha)
            tp += len(matches); fn += ng - len(matches); fp += npred - len(matches)
            gl = _arr(gt["labels"], np.int64); gi = _arr(gt["track_ids"], np.int64)
            pl = _arr(pred["labels"], np.int64); pi = _arr(pred["track_ids"], np.int64)
            for gkey, pkey, _ in matches:
                pair = (gkey, pkey)
                alpha_pairs[pair] = alpha_pairs.get(pair, 0) + 1
        deta = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        assa = (sum(count * count / (g_counts[g] + p_counts[p] - count)
                    for (g, p), count in alpha_pairs.items()) / tp) if tp else 0.0
        hota_values.append(float(np.sqrt(deta * assa)))
    matched = sum(frame_cache)
    mota = 1 - ((total_gt - matched) + (total_pred - matched) + id_switches) / total_gt if total_gt else 0.0
    
    return {
        "hota": float(np.mean(hota_values)), 
        "idf1": float(idf1), 
        "mota": float(mota),
        "id_switches": int(id_switches)
    }
