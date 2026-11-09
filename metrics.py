"""
metrics.py
==========
Evaluation metric functions for three computer vision tasks:
  1. Image classification (binary, multiclass, multilabel)
  2. Object detection
  3. Segmentation (binary, semantic, instance)

Every metric is computed "by hand" with NumPy, to show exactly what it does.
The notebooks check the results against scikit-learn.

Vocabulary:
  TP = true positive, FP = false positive, FN = false negative, TN = true negative
"""

import numpy as np


# =============================================================================
# 1. CLASSIFICATION
# =============================================================================

def binary_counts(y_true, y_pred):
    """Binary confusion matrix.

    Inputs : y_true, y_pred -- arrays of 0/1 (1 = positive).
    Output : (tp, fp, fn, tn) -- four integers.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return tp, fp, fn, tn


def accuracy(y_true, y_pred):
    """Accuracy = (TP + TN) / total. Fraction of correct answers."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(tp, fp):
    """Precision = TP / (TP + FP). "When the model says positive, is it right?" """
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp, fn):
    """Recall = TP / (TP + FN). "Of all real positives, how many were found?" """
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(tp, fp, fn):
    """F1 = 2*TP / (2*TP + FP + FN). High only if precision AND recall are both high."""
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0


def apply_threshold(scores, threshold):
    """Turns scores (between 0 and 1) into 0/1 decisions: score >= threshold -> 1."""
    return (np.asarray(scores) >= threshold).astype(int)


def roc_auc_pairs(y_true, scores):
    """ROC-AUC by counting pairs.

    AUC = probability that a random positive gets a higher score than
    a random negative (a tie counts as 1/2).
    Inputs : y_true (0/1), scores (floats). Output : AUC between 0 and 1.
    """
    y_true = np.asarray(y_true); scores = np.asarray(scores, dtype=float)
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def confusion_matrix(y_true, y_pred, n_classes):
    """Multiclass confusion matrix (N x N).

    Rows = true class, columns = predicted class, diagonal = correct answers.
    Inputs : y_true, y_pred -- integers from 0 to n_classes-1.
    """
    y_true = np.asarray(y_true).ravel().astype(int); y_pred = np.asarray(y_pred).ravel().astype(int)
    return np.bincount(n_classes * y_true + y_pred,
                       minlength=n_classes ** 2).reshape(n_classes, n_classes)


def per_class_counts(cm):
    """TP, FP, FN of each class from a confusion matrix.

    TP = diagonal, FN = rest of the row, FP = rest of the column.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return tp, fp, fn


def per_class_metrics(cm):
    """Precision, recall and F1 of each class (arrays of length N)."""
    tp, fp, fn = per_class_counts(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return p, r, f


def averaged_f1(cm, average="macro"):
    """F1 averaged over classes.

    average = "macro"    : simple average, every class counts the same
              "weighted" : average weighted by the size of each class
              "micro"    : add up TP, FP, FN of all classes, then compute one F1
    """
    tp, fp, fn = per_class_counts(cm)
    _, _, f = per_class_metrics(cm)
    if average == "macro":
        return float(f.mean())
    if average == "weighted":
        support = cm.sum(axis=1)
        return float(np.sum(f * support) / support.sum())
    if average == "micro":
        return float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()))
    raise ValueError("average must be 'macro', 'weighted' or 'micro'")


def top_k_accuracy(true_labels, ranked_guesses, k):
    """Top-k accuracy: correct if the true class is among the first k guesses.

    Inputs : true_labels -- list of true classes
             ranked_guesses -- for each image, the model's guesses,
                               from most likely to least likely
    """
    hits = [truth in guesses[:k] for truth, guesses in zip(true_labels, ranked_guesses)]
    return float(np.mean(hits))


def subset_accuracy(Y_true, Y_pred):
    """Multilabel: an image counts as correct only if ALL its labels are exactly right."""
    return float(np.mean(np.all(np.asarray(Y_true) == np.asarray(Y_pred), axis=1)))


def hamming_loss(Y_true, Y_pred):
    """Multilabel: fraction of wrong yes/no decisions (lower is better)."""
    return float(np.mean(np.asarray(Y_true) != np.asarray(Y_pred)))


def multilabel_counts(Y_true, Y_pred):
    """Multilabel: TP, FP, FN for each label (each column = one yes/no question)."""
    Y_true = np.asarray(Y_true).astype(bool); Y_pred = np.asarray(Y_pred).astype(bool)
    tp = np.sum(Y_true & Y_pred, axis=0)
    fp = np.sum(~Y_true & Y_pred, axis=0)
    fn = np.sum(Y_true & ~Y_pred, axis=0)
    return tp, fp, fn


# =============================================================================
# 2. OBJECT DETECTION
# =============================================================================

def box_iou(a, b):
    """IoU between two boxes (x_min, y_min, x_max, y_max) = intersection area / union area."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(preds, gts, iou_fn, iou_threshold=0.5):
    """Matches predictions to real objects (greedy method, as in Pascal VOC / COCO).

    Inputs : preds -- list of (box_or_mask, confidence_score)
             gts   -- list of real objects (boxes or masks)
             iou_fn -- IoU function to use (box_iou or mask_iou)
             iou_threshold -- minimum IoU for a prediction to count as correct
    Rules  : predictions are processed from most to least confident;
             each real object can be matched only once (a duplicate = FP).
    Output : list of dicts sorted by confidence, with for each prediction:
             index, score, best IoU, matched object (or None), correct (True/False).
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    matched, rows = set(), []
    for i in order:
        item, score = preds[i]
        best_iou, best_j = 0.0, None
        for j, g in enumerate(gts):
            if j in matched:
                continue
            v = iou_fn(g, item)
            if v > best_iou:
                best_iou, best_j = v, j
        correct = best_j is not None and best_iou >= iou_threshold
        if correct:
            matched.add(best_j)
        rows.append({"index": i, "score": score, "iou": best_iou,
                     "gt": best_j if correct else None, "correct": correct})
    return rows


def average_precision(rows, n_gt, interpolated=False):
    """AP with the ranked-list method.

    Read the predictions from most to least confident. At each correct prediction,
    write down (correct so far / read so far). Every real object never found counts
    as 0. AP = average of these values (one value per real object).

    interpolated=True applies the smoothing rule of official benchmarks: each written
    value is replaced by the best precision reached from that point onwards.
    """
    correct_so_far, noted, all_prec = 0, [], []
    for k, r in enumerate(rows, start=1):
        if r["correct"]:
            correct_so_far += 1
        all_prec.append(correct_so_far / k)
        if r["correct"]:
            noted.append((k - 1, correct_so_far / k))
    if interpolated:
        noted = [(pos, max(all_prec[pos:])) for pos, _ in noted]
    values = [v for _, v in noted] + [0.0] * (n_gt - len(noted))
    return float(np.mean(values)) if n_gt > 0 else 0.0


def ap_over_iou_thresholds(preds, gts, iou_fn, thresholds=None):
    """AP for several IoU thresholds (default 0.50, 0.55, ..., 0.95, as in COCO).

    Output : (thresholds, list of APs). Their average = COCO-style mAP (for one class).
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.5, 0.96, 0.05), 2)
    aps = [average_precision(match_predictions(preds, gts, iou_fn, t), len(gts)) for t in thresholds]
    return thresholds, aps


# =============================================================================
# 3. SEGMENTATION
# =============================================================================

def pixel_accuracy(gt_mask, pred_mask):
    """Pixel accuracy = correct pixels / total. Warning: fooled by the background."""
    return float(np.mean(np.asarray(gt_mask) == np.asarray(pred_mask)))


def mask_iou(gt_mask, pred_mask):
    """IoU between two binary masks = TP / (TP + FP + FN).

    "Stack the two masks": the shared area is counted once.
    """
    tp, fp, fn, _ = binary_counts(np.ravel(gt_mask), np.ravel(pred_mask))
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0


def dice(gt_mask, pred_mask):
    """Dice = 2*TP / (2*TP + FP + FN) (= F1 on pixels).

    "Put the two masks side by side": the shared area belongs to both masks,
    so it is counted twice, in the numerator and in the denominator.
    """
    tp, fp, fn, _ = binary_counts(np.ravel(gt_mask), np.ravel(pred_mask))
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 1.0


def per_class_iou(cm):
    """IoU of each class from a confusion matrix: TP / (TP + FP + FN)."""
    tp, fp, fn = per_class_counts(cm)
    return tp / (tp + fp + fn)


def mean_iou(cm):
    """mIoU = average of the per-class IoUs (every class counts the same)."""
    return float(per_class_iou(cm).mean())


def panoptic_quality(preds, gts, iou_threshold=0.5):
    """Panoptic Quality (simplified, one class): PQ = SQ x RQ.

    SQ = average IoU of the correctly found objects (shape quality)
    RQ = TP / (TP + FP/2 + FN/2) (object recognition quality)
    Inputs : preds -- list of (mask, score); gts -- list of real masks.
    Output : (PQ, SQ, RQ)
    """
    rows = match_predictions(preds, gts, mask_iou, iou_threshold)
    tp_ious = [r["iou"] for r in rows if r["correct"]]
    tp = len(tp_ious); fp = len(preds) - tp; fn = len(gts) - tp
    sq = float(np.mean(tp_ious)) if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    return sq * rq, sq, rq
