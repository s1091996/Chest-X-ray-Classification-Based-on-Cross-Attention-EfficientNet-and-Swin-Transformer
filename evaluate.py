import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score
)
from tqdm import tqdm
import pandas as pd
from config import LABEL_COLS, DEVICE

def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1

def evaluate(model, loader, threshold=None, save_file_path=None, printr=False):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    n_labels = y_true.shape[1]

    if threshold is None:
        thresholds = np.zeros(n_labels, dtype=float)
        for i in range(n_labels):
            thresholds[i], _ = find_best_threshold(y_true[:, i], y_prob[:, i])
    elif np.isscalar(threshold):
        thresholds = np.full(n_labels, threshold, dtype=float)
    else:
        thresholds = np.asarray(threshold, dtype=float)
        if len(thresholds) != n_labels:
            raise ValueError(f"threshold 數量 ({len(thresholds)}) 與類別數量 ({n_labels}) 不一致")

    y_pred = (y_prob >= thresholds[np.newaxis, :]).astype(int)

    results = {}

    for i in range(n_labels):
        label = LABEL_COLS[i] if LABEL_COLS else f"label_{i}"
        y_t = y_true[:, i]
        y_p_prob = y_prob[:, i]
        y_p = y_pred[:, i]

        try:
            auc = roc_auc_score(y_t, y_p_prob)
        except ValueError:
            auc = float('nan')

        tp = int(((y_t == 1) & (y_p == 1)).sum())
        tn = int(((y_t == 0) & (y_p == 0)).sum())
        fp = int(((y_t == 0) & (y_p == 1)).sum())
        fn = int(((y_t == 1) & (y_p == 0)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results[label] = {
            'acc': float(accuracy_score(y_t, y_p)),
            'f1': float(f1_score(y_t, y_p, zero_division=0)),
            'precision': float(precision_score(y_t, y_p, zero_division=0)),
            'recall': float(recall_score(y_t, y_p, zero_division=0)),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'auc': float(auc),
            'threshold': float(thresholds[i])
        }

    results["macro_avg"] = {
        k: float(np.nanmean([results[label][k] for label in results if label != "macro_avg"]))
        for k in ['acc', 'f1', 'precision', 'recall', 'sensitivity', 'specificity', 'auc']
    }

    if printr:
        print(f"{'Label':30s} | {'Acc':>6} | {'F1':>6} | {'AUC':>6} | {'Sens':>6} | {'Spec':>6} | {'Threshold':>9}")
        print("-" * 110)

        for label in LABEL_COLS:
            m = results[label]
            print(f"{label:30s} | {m['acc']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} | {m['threshold']:.3f}")

        print("-" * 110)
        macro = results["macro_avg"]
        print(f"{'Macro Avg':30s} | {macro['acc']:.4f} | {macro['f1']:.4f} | {macro['auc']:.4f} | {macro['sensitivity']:.4f} | {macro['specificity']:.4f}")

    if save_file_path:
        save_csv(results, save_file_path)

    return results, y_true, y_pred, y_prob, thresholds

def save_csv(results, save_file_path):
    df = pd.DataFrame(results).T
    df.to_csv(save_file_path)
    print(f"Results saved to {save_file_path}")