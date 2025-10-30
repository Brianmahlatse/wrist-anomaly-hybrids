# evaluation.py
"""
Unified evaluation utilities for the wrist hybrid study.

Covers:
1. General image-level and patient-level evaluation
2. Dataset metric flattening (dict -> DataFrame)
3. McNemar test on patient-level predictions
4. Wilcoxon signed-rank test on image vs patient pairs (the 12-pair case)
5. Patient-level evaluation for MTL comparison (general vs proposed) using
   the exact same aggregation rule as the main evaluation

Design goals:
- no plotting
- no prints
- explicit patient-level aggregation (majority vote, tie-break by prob)
- reusable for MURA, Al-Huda, and zero-shot
- consistent parameter counting and timing
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Any, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    cohen_kappa_score,
    roc_auc_score,
)

from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon as scipy_wilcoxon


# ============================================================================
# 0. Shared helpers
# ============================================================================

def _get_device(device: torch.device | None = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _logits_to_prob(logits: np.ndarray) -> np.ndarray:
    # sigmoid
    return 1.0 / (1.0 + np.exp(-logits))


# ============================================================================
# 1. General evaluation (image and patient)
# ============================================================================

@torch.no_grad()
def evaluate_and_store_metrics_torch(
    model: torch.nn.Module,
    dataset,
    model_name: str = "model",
    dataset_name: str = "dataset",
    metrics_dict: Dict[str, Dict[str, Any]] | None = None,
    *,
    patient_ids=None,
    data_collator=None,
    batch_size: int = 16,
    threshold: float = 0.5,
    measure_mem_time: bool = True,
    device: torch.device | None = None,
    info_pat: bool = False,
):
    """
    Generic evaluator used for Stages 1 to 4.

    Stores:
      - image level: acc, f1, recall, kappa, auc
      - timing and memory
      - param counts
      - predictions and probs
      - patient level if patient_ids is provided
    """
    if metrics_dict is None:
        metrics_dict = {}

    device = _get_device(device)
    model.to(device)
    model.eval()

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=data_collator,
    )

    # param counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params

    # timing and mem
    if measure_mem_time and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time() if measure_mem_time else None

    autocast_ctx = torch.amp.autocast("cuda") if torch.cuda.is_available() else nullcontext()

    all_probs = []
    all_labels = []

    for batch in dl:
        if not isinstance(batch, dict):
            raise ValueError("Dataset must return a dict with keys 'pixel_values' and 'labels'.")

        X = batch["pixel_values"]
        y = batch["labels"]

        if not torch.is_floating_point(X):
            X = X.float()
        X = X.to(device, non_blocking=True)

        with autocast_ctx:
            out = model(X)

        if isinstance(out, dict) and "logits" in out:
            logits = out["logits"]
        else:
            raise ValueError("Model output must contain 'logits'.")

        logits = logits.squeeze(1).detach().cpu().numpy().ravel()
        probs = _logits_to_prob(logits)

        all_probs.append(probs)
        all_labels.append(y.detach().cpu().numpy().ravel())

    y_true = np.concatenate(all_labels).astype(int)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob > threshold).astype(int)

    elapsed = None
    mem_mb = None
    if measure_mem_time:
        elapsed = time.time() - t0
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

    # image metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    kap = cohen_kappa_score(y_true, y_pred)
    try:
        auc_img = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_img = None

    metrics_dict.setdefault(model_name, {})
    metrics_dict[model_name][dataset_name] = {
        "accuracy": float(acc),
        "f1": float(f1),
        "recall": float(rec),
        "kappa": float(kap),
        "auc": float(auc_img) if auc_img is not None else None,
        "inference_time_sec": float(elapsed) if elapsed is not None else None,
        "memory_MB": float(mem_mb) if mem_mb is not None else None,
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable_params),
        "total_params": int(total_params),
        "y_pred": y_pred.tolist(),
        "y_pred_prob": y_prob.tolist(),
    }

    # patient level (optional)
    if patient_ids is not None:
        patient_ids = np.asarray(patient_ids).ravel()
        if len(patient_ids) != len(y_true):
            raise ValueError("patient_ids must match dataset length")

        df = pd.DataFrame(
            {
                "patient_id": patient_ids,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }
        )
        g = df.groupby("patient_id", sort=False).agg(
            y_true_sum=("y_true", "sum"),
            n=("y_true", "size"),
            y_pred_sum=("y_pred", "sum"),
            prob_mean=("y_prob", "mean"),
        )

        y_true_pat = (g["y_true_sum"] > g["n"] / 2).astype(int).to_numpy()
        y_pred_pat = (g["y_pred_sum"] > g["n"] / 2).astype(int).to_numpy()

        ties = g["y_pred_sum"] == g["n"] / 2
        if ties.any():
            y_pred_pat[ties.values] = (g.loc[ties, "prob_mean"] > threshold).astype(int).values

        try:
            auc_pat = roc_auc_score(y_true_pat, g["prob_mean"].to_numpy())
        except ValueError:
            auc_pat = None

        acc_p = accuracy_score(y_true_pat, y_pred_pat)
        f1_p = f1_score(y_true_pat, y_pred_pat, zero_division=0)
        rec_p = recall_score(y_true_pat, y_pred_pat, zero_division=0)
        kap_p = cohen_kappa_score(y_true_pat, y_pred_pat)

        metrics_dict[model_name][dataset_name].update(
            {
                "pat_accuracy": float(acc_p),
                "pat_f1": float(f1_p),
                "pat_recall": float(rec_p),
                "pat_kappa": float(kap_p),
                "pat_auc": float(auc_pat) if auc_pat is not None else None,
                "patients": int(len(g)),
                "pat_y_pred": y_pred_pat.tolist(),
                "pat_prob_mean": g["prob_mean"].to_numpy().tolist(),
                "pat_ids": g.index.to_numpy().tolist(),
            }
        )

        if info_pat:
            return metrics_dict, y_true_pat, g.index.to_numpy()

    return metrics_dict


# ============================================================================
# 2. Dict -> DataFrame
# ============================================================================

def metrics_to_long_df(metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_name, dsets in metrics.items():
        for dset_name, vals in dsets.items():
            row = {"model": model_name, "dataset": dset_name}
            row.update(vals)
            rows.append(row)
    return pd.DataFrame(rows)


def metrics_to_wide_df(metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    df_long = metrics_to_long_df(metrics)
    wide = df_long.pivot(index="model", columns="dataset").sort_index(axis=1, level=0)
    wide.columns = [f"{c[0]}_{c[1]}" for c in wide.columns]
    return wide.reset_index()


# ============================================================================
# 3. McNemar on patient-level predictions
# ============================================================================

def run_mcnemar_test(
    df: pd.DataFrame,
    model1: str,
    model2: str,
    dataset: str,
    y_true_pat: np.ndarray,
) -> Tuple[list[list[int]], Any]:
    """
    df must have: model, dataset, pat_y_pred (list-like)
    y_true_pat must be in the same patient order
    """
    preds1 = df[(df["model"] == model1) & (df["dataset"] == dataset)]["pat_y_pred"].iloc[0]
    preds2 = df[(df["model"] == model2) & (df["dataset"] == dataset)]["pat_y_pred"].iloc[0]

    if isinstance(preds1, str):
        preds1 = eval(preds1)
    if isinstance(preds2, str):
        preds2 = eval(preds2)

    preds1 = np.asarray(preds1, dtype=int)
    preds2 = np.asarray(preds2, dtype=int)
    y_true_pat = np.asarray(y_true_pat, dtype=int)

    correct1 = preds1 == y_true_pat
    correct2 = preds2 == y_true_pat

    a = int(np.sum(correct1 & correct2))
    b = int(np.sum(correct1 & ~correct2))
    c = int(np.sum(~correct1 & correct2))
    d = int(np.sum(~correct1 & ~correct2))

    table = [[a, b], [c, d]]

    if (b + c) < 25:
        result = mcnemar(table, exact=True)
    else:
        result = mcnemar(table, exact=False)

    return table, result


# ============================================================================
# 4. Wilcoxon (image vs patient)
# ============================================================================

IMG_COLS = ["accuracy", "f1", "recall", "kappa", "auc"]
PAT_COLS = ["pat_accuracy", "pat_f1", "pat_recall", "pat_kappa", "pat_auc"]
METRIC_MAP = dict(zip(IMG_COLS, PAT_COLS))


def build_tidy_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rows of (model, dataset, metric, image, patient, diff)
    so that Wilcoxon can be run across (patient - image).
    """
    rows = []
    for _, r in df.iterrows():
        for m in IMG_COLS:
            pm = METRIC_MAP[m]
            if m in r and pm in r:
                rows.append(
                    {
                        "model": r["model"],
                        "dataset": r["dataset"],
                        "metric": m,
                        "image": float(r[m]),
                        "patient": float(r[pm]),
                        "diff": float(r[pm] - r[m]),
                    }
                )
    return pd.DataFrame(rows)


def wilcoxon_effects(diffs: np.ndarray) -> dict:
    diffs = np.asarray(diffs, float)
    diffs = diffs[diffs != 0.0]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "W": math.nan, "p": math.nan, "z": math.nan, "r": math.nan}

    res = scipy_wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", mode="auto")
    W = float(res.statistic)
    p = float(res.pvalue)

    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sigma
    r = abs(z) / math.sqrt(n)

    return {"n": n, "W": W, "p": p, "z": z, "r": r}


def wilcoxon_overall_by_metric(tidy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, grp in tidy.groupby("metric"):
        stats = wilcoxon_effects(grp["diff"].to_numpy())
        rows.append(
            {
                "metric": metric,
                "n_pairs": stats["n"],
                "median_diff": float(np.median(grp["diff"])) if stats["n"] > 0 else float("nan"),
                "mean_diff": float(np.mean(grp["diff"])) if stats["n"] > 0 else float("nan"),
                "W_statistic": stats["W"],
                "z": stats["z"],
                "p_value_two_sided": stats["p"],
                "effect_size_r": stats["r"],
            }
        )
    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


# ============================================================================
# 5. MTL patient-level evaluation (general vs proposed)
# ============================================================================

def prepare_patient_truth(patient_ids, y_true):
    """
    Build patient-level ground truth once per dataset.
    Majority vote on labels.
    """
    patient_ids = np.asarray(patient_ids).ravel()
    y_true = np.asarray(y_true).ravel().astype(int)
    df = pd.DataFrame({"patient_id": patient_ids, "y_true": y_true})
    g = df.groupby("patient_id", sort=False).agg(
        y_true_sum=("y_true", "sum"),
        n=("y_true", "size"),
    )
    patient_order = g.index.to_numpy()
    y_true_pat = (g["y_true_sum"] > g["n"] / 2).astype(int).to_numpy()
    return patient_order, y_true_pat


@torch.no_grad()
def evaluate_patient_level_mtl(
    model: torch.nn.Module,
    dataset,
    sample_patient_ids,
    patient_order,
    y_true_pat,
    *,
    model_name: str,
    split_name: str,
    mtl_type: str,
    batch_size: int = 16,
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> dict:
    """
    Evaluation used specifically for the general vs proposed MTL table.
    Same aggregation rule as everywhere else.
    """
    device = _get_device(device)
    model.to(device)
    model.eval()

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    autocast_ctx = torch.amp.autocast("cuda") if torch.cuda.is_available() else nullcontext()

    probs_all = []

    for batch in dl:
        if isinstance(batch, dict):
            X = batch["pixel_values"]
        else:
            X = batch[0]

        if not torch.is_floating_point(X):
            X = X.float()
        X = X.to(device, non_blocking=True)

        with autocast_ctx:
            out = model(X)

        logits = out["logits"] if isinstance(out, dict) else out
        logits = logits.squeeze(1).detach().cpu().numpy().ravel()
        probs_all.append(_logits_to_prob(logits))

    y_prob_img = np.concatenate(probs_all)
    y_pred_img = (y_prob_img > threshold).astype(int)

    dfp = pd.DataFrame(
        {
            "patient_id": np.asarray(sample_patient_ids).ravel(),
            "y_pred": y_pred_img,
            "y_prob": y_prob_img,
        }
    )
    g = dfp.groupby("patient_id", sort=False).agg(
        y_pred_sum=("y_pred", "sum"),
        n=("y_pred", "size"),
        prob_mean=("y_prob", "mean"),
    )
    # align to precomputed order
    g = g.loc[patient_order]

    y_pred_pat = (g["y_pred_sum"] > g["n"] / 2).astype(int).to_numpy()
    ties = g["y_pred_sum"] == g["n"] / 2
    if ties.any():
        y_pred_pat[ties.values] = (g.loc[ties, "prob_mean"] > threshold).astype(int).values

    try:
        auc_pat = roc_auc_score(y_true_pat, g["prob_mean"].to_numpy())
    except ValueError:
        auc_pat = None

    elapsed = time.time() - t0
    mem_mb = (
        torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        if torch.cuda.is_available()
        else None
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params

    return {
        "model": model_name,
        "mtl_type": mtl_type,
        "split": split_name,
        "patients": int(len(patient_order)),
        "pat_accuracy": float(accuracy_score(y_true_pat, y_pred_pat)),
        "pat_f1": float(f1_score(y_true_pat, y_pred_pat, zero_division=0)),
        "pat_recall": float(recall_score(y_true_pat, y_pred_pat, zero_division=0)),
        "pat_kappa": float(cohen_kappa_score(y_true_pat, y_pred_pat)),
        "pat_auc": float(auc_pat) if auc_pat is not None else None,
        "inference_time_sec": float(elapsed),
        "memory_MB": float(mem_mb) if mem_mb is not None else None,
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable_params),
        "total_params": int(total_params),
    }


def run_models_patient_level(
    models_mapping: Iterable[tuple[str, dict[str, tuple[torch.nn.Module, torch.nn.Module]]]],
    ds_registry: dict,
    *,
    batch_size: int = 16,
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """
    models_mapping example:
        [
          (
            "XC-DEIT",
            {
              "MURA": (proposed_model, general_model),
              "Al-Huda": (proposed_model, general_model),
              "ZS-Al-Huda": (proposed_model, general_model),
            }
          ),
          ...
        ]

    ds_registry example:
        {
          "MURA": {
             "ds": mura_test_dataset,
             "sample_pids": mura_patient_ids,
             "order": mura_patient_order,
             "y_true_pat": mura_y_true_pat
          },
          "Al-Huda": {...},
          "ZS-Al-Huda": {...}
        }
    """
    rows = []
    for model_label, split_map in models_mapping:
        for split_name, (proposed_model, general_model) in split_map.items():
            if split_name not in ds_registry:
                raise ValueError(f"Unknown split {split_name} in ds_registry.")

            ds_info = ds_registry[split_name]

            # general MTL
            res_gen = evaluate_patient_level_mtl(
                model=general_model,
                dataset=ds_info["ds"],
                sample_patient_ids=ds_info["sample_pids"],
                patient_order=ds_info["order"],
                y_true_pat=ds_info["y_true_pat"],
                model_name="GEN_" + model_label,
                split_name=split_name,
                mtl_type="general",
                batch_size=batch_size,
                threshold=threshold,
                device=device,
            )
            rows.append(res_gen)

            # proposed MTL
            res_prop = evaluate_patient_level_mtl(
                model=proposed_model,
                dataset=ds_info["ds"],
                sample_patient_ids=ds_info["sample_pids"],
                patient_order=ds_info["order"],
                y_true_pat=ds_info["y_true_pat"],
                model_name=model_label,
                split_name=split_name,
                mtl_type="proposed",
                batch_size=batch_size,
                threshold=threshold,
                device=device,
            )
            rows.append(res_prop)

    return pd.DataFrame(rows)
