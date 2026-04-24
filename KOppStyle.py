# ============================================================
This Project developed and designed by Pegah Ahadian on April 2026.
# ============================================================
# ============================================================
# DL-MO: Binary detector + grouped AUC and d'
# ------------------------------------------------------------
# Primary design:
#   Train:    IMAR Full only
#   Validate: IMAR Full only
#   Test:     FBP Full + FBP Half only
#
# For each lesion × modality × dose group, compute:
#   - model score distributions
#   - model AUC
#   - model d'
# Compare:
#   - model d'  vs human mean confidence
#   - model d'  vs human detection rate
#   - model AUC vs human mean confidence
#   - model AUC vs human detection rate
#
# Outputs per run:
#   - split_manifest_units.csv
#   - split_manifest_slices.csv
#   - best_model.pt
#   - train_curve.csv
#   - train_curve.png
#   - test_slice_predictions.csv
#   - test_set_aggregates.csv
#   - test_lesion_method_dose_aggregates.csv
#   - subgroup_metrics_fbp.csv
#   - human_alignment.csv
#   - run_summary.json
#
# Global outputs:
#   - all_runs_summary.csv
#   - top10_by_mapd_confidence.csv
#   - top10_average_metrics.json
#   - experiment_readme.md
# ============================================================

import os
import re
import json
import math
import copy
import time
import random
import argparse
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================
DEFAULT_DATA_DIR = "./data/MO_CHO_Lesion_png_16bit_allSlices"
DEFAULT_OUT_DIR = "./results/dlmo_grouped_auc_dprime"
IMG_SIZE = (136, 136)
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 8
DEFAULT_EXCLUDE_MARGIN = 4
DEFAULT_NUM_RUNS = 20
DEFAULT_BASE_SEED = 42
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FNAME_RE = re.compile(
    r"^(?P<signal>SP|SA)_"
    r"(?P<lesion>Lesion\d{2})_"
    r"(?P<loc>Loc\d{2})_"
    r"(?P<dose>[FH])_"
    r"(?P<method>FBP|FPB|IMAR)_"
    r"(?P<setid>\d{3})_"
    r"slice(?P<slice_idx>\d+)_of_(?P<slice_total>\d+)\.png$",
    re.IGNORECASE
)


# ============================================================
# Human observer reference
# LG No Threshold table from the memo, mapped as:
#   Half -> low dose (150)
#   Full -> clinical dose (300)
# ============================================================
def build_human_reference_df():
    rows = []

    fbp_half_detection = {
        1: 100.0, 2: 100.0, 3: 91.7, 4: 8.3,
        5: 83.3, 6: 58.3, 7: 91.7, 8: 83.3,
        9: 0.0, 10: 8.3, 11: 0.0, 12: 0.0,
        13: 25.0, 14: 91.7, 15: 83.3, 16: 100.0,
    }
    fbp_half_confidence = {
        1: 93.3, 2: 92.5, 3: 77.5, 4: 15.0,
        5: 67.5, 6: 59.2, 7: 70.8, 8: 63.3,
        9: 36.7, 10: 34.2, 11: 15.8, 12: 32.5,
        13: 35.0, 14: 88.3, 15: 74.2, 16: 91.7,
    }

    fbp_full_detection = {
        1: 100.0, 2: 100.0, 3: 100.0, 4: 41.7,
        5: 100.0, 6: 75.0, 7: 100.0, 8: 100.0,
        9: 25.0, 10: 0.0, 11: 0.0, 12: 33.3,
        13: 50.0, 14: 91.7, 15: 83.3, 16: 100.0,
    }
    fbp_full_confidence = {
        1: 100.0, 2: 97.5, 3: 88.3, 4: 28.3,
        5: 85.8, 6: 64.2, 7: 90.0, 8: 88.3,
        9: 40.0, 10: 33.3, 11: 12.5, 12: 36.7,
        13: 43.3, 14: 89.2, 15: 82.5, 16: 95.8,
    }

    for lesion_num in range(1, 17):
        rows.append({
            "LesionNum": lesion_num,
            "Method": "FBP",
            "Dose": "Half",
            "HumanDetectionRatePct": fbp_half_detection[lesion_num],
            "HumanMeanConfidence": fbp_half_confidence[lesion_num],
            "HumanGroupKey": f"Lesion{lesion_num:02d}_FBP_Half",
        })
        rows.append({
            "LesionNum": lesion_num,
            "Method": "FBP",
            "Dose": "Full",
            "HumanDetectionRatePct": fbp_full_detection[lesion_num],
            "HumanMeanConfidence": fbp_full_confidence[lesion_num],
            "HumanGroupKey": f"Lesion{lesion_num:02d}_FBP_Full",
        })

    return pd.DataFrame(rows).sort_values(["LesionNum", "Method", "Dose"]).reset_index(drop=True)


HUMAN_REF_DF = build_human_reference_df()


# ============================================================
# Reproducibility
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Parsing
# ============================================================
def parse_filename(fname: str):
    m = FNAME_RE.match(os.path.basename(fname))
    if not m:
        return None

    d = m.groupdict()
    method = d["method"].upper()
    if method == "FPB":
        method = "FBP"

    dose = "Full" if d["dose"].upper() == "F" else "Half"
    lesion_num = int(d["lesion"][-2:])
    label = 1 if d["signal"].upper() == "SP" else 0

    # split by acquisition unit, intentionally excluding method/dose
    acquisition_unit_key = f'{d["lesion"]}_{d["loc"]}_{d["setid"]}'

    return {
        "signal": d["signal"].upper(),
        "label": label,
        "lesion": d["lesion"],
        "lesion_num": lesion_num,
        "location": d["loc"],
        "dose": dose,
        "method": method,
        "setid": d["setid"],
        "slice_idx": int(d["slice_idx"]),
        "slice_total": int(d["slice_total"]),
        "set_key": f'{d["lesion"]}_{d["loc"]}_{dose}_{method}_{d["setid"]}',
        "acquisition_unit_key": acquisition_unit_key,
        "group_key": f'{d["lesion"]}_{method}_{dose}',
    }


# ============================================================
# Scan dataset
# ============================================================
def scan_dataset(base_dir: str, exclude_margin: int = 4):
    samples_by_set = defaultdict(list)

    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.lower().endswith(".png"):
                continue

            meta = parse_filename(fname)
            if meta is None:
                continue

            meta["fpath"] = os.path.join(root, fname)
            samples_by_set[meta["set_key"]].append(meta)

    filtered = []
    for _, items in samples_by_set.items():
        items_sorted = sorted(items, key=lambda x: x["slice_idx"])
        if not items_sorted:
            continue

        total = items_sorted[0]["slice_total"]
        keep = [
            x for x in items_sorted
            if (x["slice_idx"] > exclude_margin) and (x["slice_idx"] <= total - exclude_margin)
        ]
        filtered.extend(keep)

    return filtered


# ============================================================
# Condition filters
# ============================================================
def is_imar_full(x):
    return x["method"] == "IMAR" and x["dose"] == "Full"


def is_fbp_test(x):
    return x["method"] == "FBP" and x["dose"] in {"Full", "Half"}


# ============================================================
# Split helpers
# ============================================================
def stratified_unit_split(unit_keys, unit_to_labels, test_size, seed):
    y = []
    for u in unit_keys:
        labels = unit_to_labels[u]
        p = float(np.mean(labels))
        if p < 0.33:
            y.append(0)
        elif p > 0.66:
            y.append(2)
        else:
            y.append(1)

    try:
        a, b = train_test_split(
            unit_keys,
            test_size=test_size,
            random_state=seed,
            stratify=y
        )
    except ValueError:
        a, b = train_test_split(
            unit_keys,
            test_size=test_size,
            random_state=seed,
            shuffle=True
        )
    return a, b


def build_run_splits(all_items, seed, train_frac=0.80, val_frac=0.05, test_frac=0.15):
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.")

    test_candidate_items = [x for x in all_items if is_fbp_test(x)]
    test_candidate_units = sorted(set(x["acquisition_unit_key"] for x in test_candidate_items))
    if len(test_candidate_units) < 3:
        raise RuntimeError("Not enough FBP test candidate acquisition units.")

    unit_to_labels = defaultdict(list)
    for x in all_items:
        unit_to_labels[x["acquisition_unit_key"]].append(x["label"])

    non_test_units, test_units = stratified_unit_split(
        test_candidate_units,
        unit_to_labels=unit_to_labels,
        test_size=test_frac,
        seed=seed
    )

    trainval_items = [
        x for x in all_items
        if x["acquisition_unit_key"] in set(non_test_units) and is_imar_full(x)
    ]
    if len(trainval_items) == 0:
        raise RuntimeError("No IMAR Full items available in non-test units.")

    trainval_units = sorted(set(x["acquisition_unit_key"] for x in trainval_items))
    if len(trainval_units) < 2:
        raise RuntimeError("Not enough IMAR Full train/val units after test split.")

    val_within_trainval = val_frac / (train_frac + val_frac)

    train_units, val_units = stratified_unit_split(
        trainval_units,
        unit_to_labels=unit_to_labels,
        test_size=val_within_trainval,
        seed=seed + 17
    )

    train_items = [x for x in trainval_items if x["acquisition_unit_key"] in set(train_units)]
    val_items = [x for x in trainval_items if x["acquisition_unit_key"] in set(val_units)]
    test_items = [
        x for x in all_items
        if x["acquisition_unit_key"] in set(test_units) and is_fbp_test(x)
    ]

    if len(test_items) == 0:
        raise RuntimeError("No FBP test items selected for this run.")
    if len(train_items) == 0 or len(val_items) == 0:
        raise RuntimeError("Train or validation set is empty.")

    assert all(is_imar_full(x) for x in train_items)
    assert all(is_imar_full(x) for x in val_items)
    assert all(is_fbp_test(x) for x in test_items)

    overlap_train_test = set(x["acquisition_unit_key"] for x in train_items) & set(x["acquisition_unit_key"] for x in test_items)
    overlap_val_test = set(x["acquisition_unit_key"] for x in val_items) & set(x["acquisition_unit_key"] for x in test_items)
    if overlap_train_test or overlap_val_test:
        raise RuntimeError("Leakage detected between train/val and test.")

    return train_items, val_items, test_items, train_units, val_units, test_units


# ============================================================
# Dataset
# ============================================================
class LesionDataset(Dataset):
    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment
        self.aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        meta = self.items[idx]
        img = cv2.imread(meta["fpath"], cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Could not read image: {meta['fpath']}")

        img = img.astype(np.float32)
        img = img / 4095.0
        img = np.clip(img, 0.0, 1.0)

        if self.augment:
            alpha = np.random.uniform(0.90, 1.10)
            beta = np.random.uniform(-0.08, 0.08)
            img = np.clip(alpha * img + beta, 0.0, 1.0)

        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        img = (img - 0.5) / 0.5
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        if self.augment:
            img = self.aug(img)

        fname = os.path.basename(meta["fpath"])
        return img, torch.tensor(meta["label"], dtype=torch.float32), fname


# ============================================================
# Kopp-like shallow CNN, binary score-producing detector
# ============================================================
class KoppLikeBinaryCNN(nn.Module):
    def __init__(self, input_shape=(1, 136, 136)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            flat_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x).squeeze(1)
        return x


# ============================================================
# Metrics
# ============================================================
def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float32)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if len(y_true) == 0:
        return {"AUC": np.nan, "ACC": np.nan, "SENS": np.nan, "SPEC": np.nan, "F1": np.nan, "N": 0}

    y_pred = (y_prob >= threshold).astype(np.int32)
    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = safe_auc(y_true, y_prob)

    return {
        "AUC": auc,
        "ACC": float(acc),
        "SENS": float(sens),
        "SPEC": float(spec),
        "F1": float(f1),
        "N": int(len(y_true))
    }


def compute_pos_weight(train_items):
    labels = np.array([x["label"] for x in train_items], dtype=np.int32)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0:
        return torch.tensor(1.0, device=DEVICE)
    return torch.tensor(neg / max(pos, 1), dtype=torch.float32, device=DEVICE)


def pooled_std(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    va = float(np.var(a, ddof=0)) if len(a) > 0 else 0.0
    vb = float(np.var(b, ddof=0)) if len(b) > 0 else 0.0
    return math.sqrt(0.5 * (va + vb) + 1e-8)


def compute_dprime_from_scores(pos_scores, neg_scores):
    pos_scores = np.asarray(pos_scores, dtype=np.float32)
    neg_scores = np.asarray(neg_scores, dtype=np.float32)

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return np.nan

    ps = pooled_std(pos_scores, neg_scores)
    if ps <= 0:
        return np.nan

    return float((np.mean(pos_scores) - np.mean(neg_scores)) / ps)


def mean_absolute_percentage_difference(pred, true, min_denominator=1.0):
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    denom = np.clip(np.abs(true), min_denominator, None)
    return float(np.mean(np.abs(pred - true) / denom) * 100.0)


# ============================================================
# Train / evaluate
# ============================================================
def evaluate_model(model, loader):
    model.eval()
    all_logits, all_probs, all_labels, all_fnames = [], [], [], []

    with torch.no_grad():
        for imgs, labels, fnames in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            all_logits.extend(logits.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_fnames.extend(list(fnames))

    return all_logits, all_probs, all_labels, all_fnames


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


# ============================================================
# Aggregation
# ============================================================
def predictions_to_dataframe(test_items, fnames, labels, logits, probs):
    fname_to_meta = {os.path.basename(x["fpath"]): x for x in test_items}

    rows = []
    for fname, label, logit, prob in zip(fnames, labels, logits, probs):
        meta = fname_to_meta.get(fname, None)
        if meta is None:
            continue

        rows.append({
            "Filename": fname,
            "Label": int(label),
            "Logit": float(logit),
            "PredProb": float(prob),
            "PredLabel": int(prob >= 0.5),
            "Signal": meta["signal"],
            "Lesion": meta["lesion"],
            "LesionNum": int(meta["lesion_num"]),
            "Location": meta["location"],
            "Dose": meta["dose"],
            "Method": meta["method"],
            "SetID": meta["setid"],
            "SliceIdx": int(meta["slice_idx"]),
            "SliceTotal": int(meta["slice_total"]),
            "SetKey": meta["set_key"],
            "AcquisitionUnitKey": meta["acquisition_unit_key"],
            "GroupKey": meta["group_key"],
        })

    return pd.DataFrame(rows)


def aggregate_predictions(df_pred, group_cols):
    rows = []
    if len(df_pred) == 0:
        return pd.DataFrame()

    for gvals, gdf in df_pred.groupby(group_cols):
        if not isinstance(gvals, tuple):
            gvals = (gvals,)

        pos_df = gdf[gdf["Label"] == 1].copy()
        neg_df = gdf[gdf["Label"] == 0].copy()

        metrics = compute_binary_metrics(gdf["Label"].values, gdf["PredProb"].values)

        dprime = compute_dprime_from_scores(
            pos_scores=pos_df["Logit"].values,
            neg_scores=neg_df["Logit"].values
        )

        model_detection_rate_pct = np.nan
        model_confidence_pct = np.nan
        if len(pos_df) > 0:
            model_detection_rate_pct = float((pos_df["PredProb"].values >= 0.5).mean() * 100.0)
            model_confidence_pct = float(pos_df["PredProb"].mean() * 100.0)

        row = {col: val for col, val in zip(group_cols, gvals)}
        row.update({
            "NTotalSlices": int(len(gdf)),
            "NPosSlices": int(len(pos_df)),
            "NNegSlices": int(len(neg_df)),

            "MeanProbAll": float(gdf["PredProb"].mean()),
            "MeanProbPos": float(pos_df["PredProb"].mean()) if len(pos_df) > 0 else np.nan,
            "MeanProbNeg": float(neg_df["PredProb"].mean()) if len(neg_df) > 0 else np.nan,

            "MeanLogitAll": float(gdf["Logit"].mean()),
            "MeanLogitPos": float(pos_df["Logit"].mean()) if len(pos_df) > 0 else np.nan,
            "MeanLogitNeg": float(neg_df["Logit"].mean()) if len(neg_df) > 0 else np.nan,

            "ModelAUC": metrics["AUC"],
            "ModelDPrime": dprime,

            "ModelDetectionRatePct": model_detection_rate_pct,
            "ModelConfidencePct": model_confidence_pct,

            "ACC": metrics["ACC"],
            "SENS": metrics["SENS"],
            "SPEC": metrics["SPEC"],
            "F1": metrics["F1"],
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Human alignment
# ============================================================
def align_with_human(df_grouped):
    """
    Expects df_grouped aggregated at:
      LesionNum, Method, Dose
    """

    if len(df_grouped) == 0:
        return pd.DataFrame(), {
            "n_human_matched_groups": 0,

            "spearman_dprime_vs_human_confidence_r": np.nan,
            "spearman_dprime_vs_human_confidence_p": np.nan,
            "spearman_dprime_vs_human_detection_r": np.nan,
            "spearman_dprime_vs_human_detection_p": np.nan,

            "spearman_auc_vs_human_confidence_r": np.nan,
            "spearman_auc_vs_human_confidence_p": np.nan,
            "spearman_auc_vs_human_detection_r": np.nan,
            "spearman_auc_vs_human_detection_p": np.nan,

            "mapd_model_confidence_vs_human_confidence": np.nan,
            "mapd_model_detection_vs_human_detection": np.nan,
        }

    merged = df_grouped.merge(
        HUMAN_REF_DF,
        on=["LesionNum", "Method", "Dose"],
        how="inner"
    ).copy()

    if len(merged) == 0:
        return merged, {
            "n_human_matched_groups": 0,

            "spearman_dprime_vs_human_confidence_r": np.nan,
            "spearman_dprime_vs_human_confidence_p": np.nan,
            "spearman_dprime_vs_human_detection_r": np.nan,
            "spearman_dprime_vs_human_detection_p": np.nan,

            "spearman_auc_vs_human_confidence_r": np.nan,
            "spearman_auc_vs_human_confidence_p": np.nan,
            "spearman_auc_vs_human_detection_r": np.nan,
            "spearman_auc_vs_human_detection_p": np.nan,

            "mapd_model_confidence_vs_human_confidence": np.nan,
            "mapd_model_detection_vs_human_detection": np.nan,
        }

    # d' vs human confidence
    tmp = merged.dropna(subset=["ModelDPrime", "HumanMeanConfidence"]).copy()
    if len(tmp) >= 2:
        r1, p1 = spearmanr(tmp["ModelDPrime"], tmp["HumanMeanConfidence"])
    else:
        r1, p1 = np.nan, np.nan

    # d' vs human detection
    tmp = merged.dropna(subset=["ModelDPrime", "HumanDetectionRatePct"]).copy()
    if len(tmp) >= 2:
        r2, p2 = spearmanr(tmp["ModelDPrime"], tmp["HumanDetectionRatePct"])
    else:
        r2, p2 = np.nan, np.nan

    # AUC vs human confidence
    tmp = merged.dropna(subset=["ModelAUC", "HumanMeanConfidence"]).copy()
    if len(tmp) >= 2:
        r3, p3 = spearmanr(tmp["ModelAUC"], tmp["HumanMeanConfidence"])
    else:
        r3, p3 = np.nan, np.nan

    # AUC vs human detection
    tmp = merged.dropna(subset=["ModelAUC", "HumanDetectionRatePct"]).copy()
    if len(tmp) >= 2:
        r4, p4 = spearmanr(tmp["ModelAUC"], tmp["HumanDetectionRatePct"])
    else:
        r4, p4 = np.nan, np.nan

    mapd_conf = np.nan
    tmp = merged.dropna(subset=["ModelConfidencePct", "HumanMeanConfidence"]).copy()
    if len(tmp) > 0:
        mapd_conf = mean_absolute_percentage_difference(
            pred=tmp["ModelConfidencePct"].values,
            true=tmp["HumanMeanConfidence"].values,
            min_denominator=1.0
        )

    mapd_det = np.nan
    tmp = merged.dropna(subset=["ModelDetectionRatePct", "HumanDetectionRatePct"]).copy()
    if len(tmp) > 0:
        mapd_det = mean_absolute_percentage_difference(
            pred=tmp["ModelDetectionRatePct"].values,
            true=tmp["HumanDetectionRatePct"].values,
            min_denominator=1.0
        )

    summary = {
        "n_human_matched_groups": int(len(merged)),

        "spearman_dprime_vs_human_confidence_r": float(r1) if pd.notna(r1) else np.nan,
        "spearman_dprime_vs_human_confidence_p": float(p1) if pd.notna(p1) else np.nan,
        "spearman_dprime_vs_human_detection_r": float(r2) if pd.notna(r2) else np.nan,
        "spearman_dprime_vs_human_detection_p": float(p2) if pd.notna(p2) else np.nan,

        "spearman_auc_vs_human_confidence_r": float(r3) if pd.notna(r3) else np.nan,
        "spearman_auc_vs_human_confidence_p": float(p3) if pd.notna(p3) else np.nan,
        "spearman_auc_vs_human_detection_r": float(r4) if pd.notna(r4) else np.nan,
        "spearman_auc_vs_human_detection_p": float(p4) if pd.notna(p4) else np.nan,

        "mapd_model_confidence_vs_human_confidence": float(mapd_conf) if pd.notna(mapd_conf) else np.nan,
        "mapd_model_detection_vs_human_detection": float(mapd_det) if pd.notna(mapd_det) else np.nan,
    }
    return merged, summary


# ============================================================
# Utility writers
# ============================================================
def save_split_manifests(run_dir, train_items, val_items, test_items, train_units, val_units, test_units):
    unit_rows = []
    for u in train_units:
        unit_rows.append({"Split": "train", "AcquisitionUnitKey": u})
    for u in val_units:
        unit_rows.append({"Split": "val", "AcquisitionUnitKey": u})
    for u in test_units:
        unit_rows.append({"Split": "test", "AcquisitionUnitKey": u})
    pd.DataFrame(unit_rows).sort_values(["Split", "AcquisitionUnitKey"]).to_csv(
        os.path.join(run_dir, "split_manifest_units.csv"),
        index=False
    )

    slice_rows = []
    for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        for x in items:
            slice_rows.append({
                "Split": split_name,
                "Filename": os.path.basename(x["fpath"]),
                "Signal": x["signal"],
                "Label": x["label"],
                "Lesion": x["lesion"],
                "LesionNum": x["lesion_num"],
                "Location": x["location"],
                "Dose": x["dose"],
                "Method": x["method"],
                "SetID": x["setid"],
                "SliceIdx": x["slice_idx"],
                "SliceTotal": x["slice_total"],
                "SetKey": x["set_key"],
                "AcquisitionUnitKey": x["acquisition_unit_key"],
                "FPath": x["fpath"],
            })
    pd.DataFrame(slice_rows).to_csv(
        os.path.join(run_dir, "split_manifest_slices.csv"),
        index=False
    )


def write_experiment_readme(out_dir, args, dataset_items):
    text = f"""
# DL-MO grouped AUC and d-prime experiment

## Design
- Train: IMAR Full only
- Validate: IMAR Full only
- Test: FBP Full + FBP Half only
- Number of repeated runs: {args.num_runs}

## Leakage control
- Split unit: lesion + location + setid
- The same acquisition unit is never shared between train/val and test
- Test units are chosen first
- Train/val are built only from remaining IMAR Full units

## Model
- Kopp-like shallow CNN:
  - Conv(1->6, 5x5), MaxPool
  - Conv(6->16, 5x5), MaxPool
  - FC(120), FC(84), Output(1)
- Binary lesion-present vs lesion-absent classification
- Observer score = model logit

## Grouped observer metrics
For each lesion × modality × dose group:
- ModelAUC
- ModelDPrime
- ModelDetectionRatePct
- ModelConfidencePct

## Human alignment
Compare:
- ModelDPrime vs HumanMeanConfidence
- ModelDPrime vs HumanDetectionRatePct
- ModelAUC vs HumanMeanConfidence
- ModelAUC vs HumanDetectionRatePct

## Ranking of runs
Top 10 runs are selected by the lowest:
- mapd_model_confidence_vs_human_confidence

## Dataset size after slice-margin exclusion
{len(dataset_items)} slices

## Command
{vars(args)}
"""
    with open(os.path.join(out_dir, "experiment_readme.md"), "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


# ============================================================
# One run
# ============================================================
def run_one_experiment(run_idx, seed, all_items, args):
    run_name = f"run_{run_idx:02d}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    seed_everything(seed)

    train_items, val_items, test_items, train_units, val_units, test_units = build_run_splits(
        all_items=all_items,
        seed=seed,
        train_frac=0.80,
        val_frac=0.05,
        test_frac=0.15,
    )

    save_split_manifests(run_dir, train_items, val_items, test_items, train_units, val_units, test_units)

    train_ds = LesionDataset(train_items, augment=True)
    val_ds = LesionDataset(val_items, augment=False)
    test_ds = LesionDataset(test_items, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model = KoppLikeBinaryCNN(input_shape=(1, IMG_SIZE[0], IMG_SIZE[1])).to(DEVICE)

    pos_weight = compute_pos_weight(train_items)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_state = None
    best_val_auc = -np.inf
    best_epoch = -1
    no_improve = 0
    curve_rows = []

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        _, val_probs, val_labels, _ = evaluate_model(model, val_loader)
        val_metrics = compute_binary_metrics(val_labels, val_probs)

        val_auc = val_metrics["AUC"]
        if np.isnan(val_auc):
            val_auc = 0.0

        scheduler.step(val_auc)

        curve_rows.append({
            "Epoch": epoch,
            "TrainLoss": train_loss,
            "ValAUC": val_metrics["AUC"],
            "ValACC": val_metrics["ACC"],
            "ValSENS": val_metrics["SENS"],
            "ValSPEC": val_metrics["SPEC"],
            "ValF1": val_metrics["F1"],
            "LR": optimizer.param_groups[0]["lr"],
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            break

    train_time_sec = time.time() - t0

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(os.path.join(run_dir, "train_curve.csv"), index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(curve_df["Epoch"], curve_df["TrainLoss"], label="Train Loss")
    plt.plot(curve_df["Epoch"], curve_df["ValAUC"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"{run_name}: IMAR Full train/val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "train_curve.png"), dpi=180)
    plt.close()

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(run_dir, "best_model.pt"))

    # Test predictions
    test_logits, test_probs, test_labels, test_fnames = evaluate_model(model, test_loader)
    df_pred = predictions_to_dataframe(
        test_items=test_items,
        fnames=test_fnames,
        labels=test_labels,
        logits=test_logits,
        probs=test_probs
    )
    df_pred.to_csv(os.path.join(run_dir, "test_slice_predictions.csv"), index=False)

    # Per-set aggregate
    df_set = aggregate_predictions(
        df_pred,
        group_cols=["LesionNum", "Lesion", "Location", "Method", "Dose", "SetID", "SetKey"]
    )
    df_set.to_csv(os.path.join(run_dir, "test_set_aggregates.csv"), index=False)

    # Per lesion × modality × dose aggregate
    df_group = aggregate_predictions(
        df_pred,
        group_cols=["LesionNum", "Lesion", "Method", "Dose"]
    ).sort_values(["LesionNum", "Method", "Dose"]).reset_index(drop=True)
    df_group.to_csv(os.path.join(run_dir, "test_lesion_method_dose_aggregates.csv"), index=False)

    # Human alignment
    df_human_align, human_summary = align_with_human(df_group)
    df_human_align.to_csv(os.path.join(run_dir, "human_alignment.csv"), index=False)

    # FBP subgroup metrics
    subgroup_rows = []
    for dose in ["Full", "Half"]:
        mask = (df_pred["Method"] == "FBP") & (df_pred["Dose"] == dose)
        sub = df_pred.loc[mask].copy()
        metrics = compute_binary_metrics(sub["Label"].values, sub["PredProb"].values)
        subgroup_rows.append({"Subset": f"FBP_{dose}", **metrics})
    df_subgroup = pd.DataFrame(subgroup_rows)
    df_subgroup.to_csv(os.path.join(run_dir, "subgroup_metrics_fbp.csv"), index=False)

    overall_metrics = compute_binary_metrics(df_pred["Label"].values, df_pred["PredProb"].values)

    test_count_by_dose = df_pred.groupby(["Dose"]).size().to_dict()

    run_summary = {
        "run_idx": run_idx,
        "seed": seed,

        "train_slices": int(len(train_items)),
        "val_slices": int(len(val_items)),
        "test_slices": int(len(test_items)),

        "train_units": int(len(train_units)),
        "val_units": int(len(val_units)),
        "test_units": int(len(test_units)),

        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc) if np.isfinite(best_val_auc) else np.nan,
        "train_time_sec": float(train_time_sec),

        "test_auc_all": overall_metrics["AUC"],
        "test_acc_all": overall_metrics["ACC"],
        "test_sens_all": overall_metrics["SENS"],
        "test_spec_all": overall_metrics["SPEC"],
        "test_f1_all": overall_metrics["F1"],

        "test_auc_fbp_full": float(df_subgroup.loc[df_subgroup["Subset"] == "FBP_Full", "AUC"].iloc[0]),
        "test_acc_fbp_full": float(df_subgroup.loc[df_subgroup["Subset"] == "FBP_Full", "ACC"].iloc[0]),
        "test_sens_fbp_full": float(df_subgroup.loc[df_subgroup["Subset"] == "FBP_Full", "SENS"].iloc[0]),
        "test_spec_fbp_full": float(df_subgroup.loc[df_subgroup["Subset"] == "FBP_Full", "SPEC"].iloc[0]),
        "test_f1_fbp_full": float(df_subgroup.loc[df_subgroup["Subset"] == "FBP_Full", "F1"].iloc[0]),

        "test_auc_fbp_half": float(df_subgroup.loc[df_subgroup["Subset"] == "FBP_Half", "AUC"].iloc[0]),
        "test_acc_fbp_half": float(df_subgroup.loc[df_subgroup["Subset"] == "FBP_Half", "ACC"].iloc[0]),
        "test_sens_fbp_half": float(df_subgroup.loc[df_subgroup["Subset"] == "FBP_Half", "SENS"].iloc[0]),
        "test_spec_fbp_half": float(df_subgroup.loc[df_subgroup["Subset"] == "FBP_Half", "SPEC"].iloc[0]),
        "test_f1_fbp_half": float(df_subgroup.loc[df_subgroup["Subset"] == "FBP_Half", "F1"].iloc[0]),

        **human_summary,

        "test_count_fbp_full": int(test_count_by_dose.get("Full", 0)),
        "test_count_fbp_half": int(test_count_by_dose.get("Half", 0)),
        "n_group_rows_lesion_method_dose": int(len(df_group)),
        "n_set_rows": int(len(df_set)),
        "n_unique_test_lesions": int(df_pred["LesionNum"].nunique()),
    }

    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    return run_summary


# ============================================================
# Top 10 summary
# ============================================================
def summarize_top10(df_runs, out_dir):
    sort_col = "mapd_model_confidence_vs_human_confidence"
    top10 = df_runs.sort_values(sort_col, ascending=True).head(10).reset_index(drop=True)
    top10.to_csv(os.path.join(out_dir, "top10_by_mapd_confidence.csv"), index=False)

    numeric_cols = [
        c for c in top10.columns
        if pd.api.types.is_numeric_dtype(top10[c]) and c not in {"run_idx", "seed"}
    ]

    summary = {
        "selection_metric": sort_col,
        "n_runs_total": int(len(df_runs)),
        "n_runs_top10": int(len(top10)),
        "top10_run_indices": top10["run_idx"].tolist(),
        "means": {},
        "stds": {},
    }

    for c in numeric_cols:
        summary["means"][c] = float(top10[c].mean()) if len(top10[c]) > 0 else np.nan
        summary["stds"][c] = float(top10[c].std(ddof=1)) if len(top10[c]) > 1 else 0.0

    with open(os.path.join(out_dir, "top10_average_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return top10, summary


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--exclude_margin", type=int, default=DEFAULT_EXCLUDE_MARGIN)
    parser.add_argument("--num_runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--base_seed", type=int, default=DEFAULT_BASE_SEED)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.base_seed)

    all_items = scan_dataset(args.data_dir, exclude_margin=args.exclude_margin)
    if len(all_items) == 0:
        raise RuntimeError("No usable PNG slices found. Check data_dir and naming format.")

    dataset_rows = []
    for x in all_items:
        dataset_rows.append({
            "Method": x["method"],
            "Dose": x["dose"],
            "Signal": x["signal"],
            "LesionNum": x["lesion_num"],
            "SetID": x["setid"],
            "AcquisitionUnitKey": x["acquisition_unit_key"]
        })
    pd.DataFrame(dataset_rows).to_csv(
        os.path.join(args.out_dir, "dataset_inventory_after_margin_filter.csv"),
        index=False
    )

    write_experiment_readme(args.out_dir, args, all_items)

    print(f"Device: {DEVICE}")
    print(f"Scanned usable slices: {len(all_items)}")
    print(f"Output dir: {args.out_dir}")

    all_run_summaries = []

    for run_idx in range(1, args.num_runs + 1):
        seed = args.base_seed + run_idx - 1
        print("\n" + "=" * 70)
        print(f"Run {run_idx:02d}/{args.num_runs} | seed={seed}")
        print("=" * 70)

        run_summary = run_one_experiment(
            run_idx=run_idx,
            seed=seed,
            all_items=all_items,
            args=args
        )
        all_run_summaries.append(run_summary)

        print(
            f"best_epoch={run_summary['best_epoch']} | "
            f"val_auc={run_summary['best_val_auc']:.4f} | "
            f"test_auc_all={run_summary['test_auc_all']:.4f} | "
            f"FBP_Full_AUC={run_summary['test_auc_fbp_full']:.4f} | "
            f"FBP_Half_AUC={run_summary['test_auc_fbp_half']:.4f} | "
            f"d'_conf_r={run_summary['spearman_dprime_vs_human_confidence_r']:.3f} | "
            f"d'_det_r={run_summary['spearman_dprime_vs_human_detection_r']:.3f} | "
            f"auc_conf_r={run_summary['spearman_auc_vs_human_confidence_r']:.3f} | "
            f"auc_det_r={run_summary['spearman_auc_vs_human_detection_r']:.3f}"
        )

    df_runs = pd.DataFrame(all_run_summaries).sort_values("run_idx").reset_index(drop=True)
    df_runs.to_csv(os.path.join(args.out_dir, "all_runs_summary.csv"), index=False)

    summarize_top10(df_runs, args.out_dir)

    print("\nDone.")
    print(f"Saved all run summaries to: {os.path.join(args.out_dir, 'all_runs_summary.csv')}")
    print(f"Saved top-10 file to:       {os.path.join(args.out_dir, 'top10_by_mapd_confidence.csv')}")
    print(f"Saved top-10 summary to:    {os.path.join(args.out_dir, 'top10_average_metrics.json')}")


if __name__ == "__main__":
    main()
