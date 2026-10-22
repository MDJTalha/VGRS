#!/usr/bin/env python3
"""Report classification metrics from a labeled held-out prediction CSV.

Required columns default to true_label and predicted_label. Use --prob-prefix
when calibrated class score columns are available, e.g. prob_child, prob_feminine.
Do not treat arbitrary softmax scores as calibrated probabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, log_loss, precision_recall_fscore_support,
)


ABSTAIN = {"", "uncertain", "insufficient evidence", "insufficient_evidence", "abstain", "unknown", "model_unavailable"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None, help="JSON destination (prints to stdout by default)")
    p.add_argument("--true-col", default="true_label")
    p.add_argument("--pred-col", default="predicted_label")
    p.add_argument("--prob-prefix", default="prob_", help="Prefix of one probability column per class")
    p.add_argument("--subgroup-col", default=None, help="Optional demographic/accent/condition column for subgroup metrics")
    p.add_argument("--speaker-col", default="speaker_id", help="Used to report held-out speaker count when present")
    p.add_argument("--bootstrap", type=int, default=0, help="Speaker-cluster bootstrap replicates for macro-F1 CI; 0 disables")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def is_abstain(value: Any) -> bool:
    return str(value).strip().lower() in ABSTAIN


def core_metrics(true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    covered = ~pred.map(is_abstain)
    yt = true[covered].astype(str)
    yp = pred[covered].astype(str)
    if len(yt) == 0:
        return {"n": int(len(true)), "n_covered": 0, "coverage": 0.0,
                "accuracy": None, "balanced_accuracy": None, "macro_f1": None,
                "per_class": {}, "confusion_matrix": {"labels": [], "matrix": []}}
    labels = sorted(set(true.astype(str)) | set(yp.astype(str)))
    report = classification_report(yt, yp, labels=labels, output_dict=True, zero_division=0)
    cm_labels = sorted(set(yt.astype(str)) | set(yp.astype(str)))
    cm = confusion_matrix(yt, yp, labels=cm_labels)
    per_class = {label: {
        "precision": float(report[label]["precision"]),
        "recall": float(report[label]["recall"]),
        "f1": float(report[label]["f1-score"]),
        "support": int(report[label]["support"]),
    } for label in labels if label in report}
    return {
        "n": int(len(true)), "n_covered": int(covered.sum()),
        "n_abstained": int((~covered).sum()), "coverage": float(covered.mean()),
        "accuracy_on_covered": float(accuracy_score(yt, yp)),
        "balanced_accuracy_on_covered": float(balanced_accuracy_score(yt, yp)),
        "macro_f1_on_covered": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": {"labels": cm_labels, "matrix": cm.tolist()},
    }


def probabilities_metrics(frame: pd.DataFrame, labels: list[str], prefix: str) -> dict[str, Any] | None:
    cols = [prefix + label for label in labels]
    if not all(col in frame.columns for col in cols):
        return None
    probs = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        return {"available": False, "reason": "Scores must be finite values between 0 and 1."}
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.02):
        return {"available": False, "reason": "Rows must sum to approximately 1.0; do not pass arbitrary logits."}
    true = frame["__true"].astype(str).to_numpy()
    if not set(true).issubset(set(labels)):
        return {"available": False, "reason": "True labels do not match probability column classes."}
    class_to_idx = {label: index for index, label in enumerate(labels)}
    target_idx = np.array([class_to_idx[v] for v in true])
    one_hot = np.eye(len(labels))[target_idx]
    clipped = np.clip(probs, 1e-12, 1 - 1e-12)
    one_vs_rest = [brier_score_loss(one_hot[:, i], probs[:, i]) for i in range(len(labels))]
    confidences = probs.max(axis=1)
    hits = (np.argmax(probs, axis=1) == target_idx).astype(float)
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi if hi < 1 else confidences <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(hits[mask].mean()) - float(confidences[mask].mean()))
    return {
        "available": True, "class_order": labels, "multiclass_brier_mean_ovr": float(np.mean(one_vs_rest)),
        "log_loss": float(log_loss(true, clipped, labels=labels)),
        "expected_calibration_error_10_bins": float(ece),
        "calibration_note": "Computed on this CSV only; report these metrics on a held-out calibration/evaluation split, not training predictions.",
    }


def bootstrap_macro_f1(frame: pd.DataFrame, speaker_col: str, replicates: int, seed: int) -> dict[str, Any] | None:
    if replicates <= 0 or speaker_col not in frame.columns:
        return None
    speaker = frame[speaker_col].astype(str)
    if (speaker.str.strip() == "").any():
        return {"available": False, "reason": "Empty speaker IDs prevent cluster bootstrap."}
    groups = list(speaker.unique())
    if len(groups) < 2:
        return {"available": False, "reason": "At least two independent speakers are required."}
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    by_speaker = {g: frame.index[speaker == g].to_numpy() for g in groups}
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([by_speaker[g] for g in sampled])
        sample = frame.loc[idx]
        covered = ~sample["__pred"].map(is_abstain)
        sample = sample.loc[covered]
        if len(sample) and sample["__true"].nunique() >= 2:
            scores.append(float(f1_score(sample["__true"].astype(str), sample["__pred"].astype(str), average="macro", zero_division=0)))
    if not scores:
        return {"available": False, "reason": "No valid bootstrap replicates remained after abstention filtering."}
    return {"available": True, "replicates_requested": replicates, "replicates_valid": len(scores),
            "macro_f1_ci_95_percentile": [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))],
            "unit": "speaker cluster"}


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.predictions, dtype=str, keep_default_na=False)
    for col in (args.true_col, args.pred_col):
        if col not in frame.columns:
            raise SystemExit(f"Required column not found: {col}")
    if frame.empty:
        raise SystemExit("Prediction file contains no rows.")
    frame = frame.copy()
    frame["__true"] = frame[args.true_col].astype(str)
    frame["__pred"] = frame[args.pred_col].astype(str)
    if (frame["__true"].str.strip() == "").any():
        raise SystemExit("Ground-truth labels must not be empty.")
    result: dict[str, Any] = {
        "source_file": str(args.predictions),
        "n_rows": int(len(frame)),
        "target": args.true_col,
        "prediction": args.pred_col,
        "primary_metrics": core_metrics(frame["__true"], frame["__pred"]),
        "probability_metrics": None,
        "speaker_bootstrap": bootstrap_macro_f1(frame, args.speaker_col, args.bootstrap, args.seed),
        "subgroups": {},
        "limitations": [
            "These metrics describe only the provided prediction file; data quality and split independence must be audited separately.",
            "Abstentions are excluded from accuracy/balanced-accuracy/macro-F1 and reported as coverage; always report coverage with selective metrics.",
            "Probability metrics require class-probability columns that sum to approximately one and must be computed on held-out predictions.",
            "Subgroup comparisons can be unstable for small samples; report denominators and uncertainty.",
        ],
    }
    labels = sorted(frame["__true"].astype(str).unique())
    result["probability_metrics"] = probabilities_metrics(frame, labels, args.prob_prefix)
    if args.subgroup_col:
        if args.subgroup_col not in frame.columns:
            raise SystemExit(f"Subgroup column not found: {args.subgroup_col}")
        for value, subset in frame.groupby(args.subgroup_col, dropna=False):
            if str(value).strip():
                result["subgroups"][str(value)] = core_metrics(subset["__true"], subset["__pred"])
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
