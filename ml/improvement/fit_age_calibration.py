from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def age_group(age_years: float) -> str:
    if age_years < 13:
        return "Under 13"
    if age_years < 18:
        return "13–17"
    if age_years < 30:
        return "18–29"
    if age_years < 45:
        return "30–44"
    if age_years < 60:
        return "45–59"
    return "60+"


def mae(rows: list[dict[str, Any]], key: str) -> float:
    return mean(abs(float(row[key]) - float(row["actual_age_years"])) for row in rows)


def group_accuracy(rows: list[dict[str, Any]], key: str) -> float:
    return mean(age_group(float(row[key])) == age_group(float(row["actual_age_years"])) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit an experimental age calibration on speaker-separated data and evaluate on held-out speakers."
    )
    parser.add_argument("--predictions", type=Path, default=ROOT / "reports" / "research-improvement" / "predictions.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "research-improvement" / "age-calibration-candidate.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-speakers", type=int, default=30)
    args = parser.parse_args()

    if args.min_speakers < 30:
        raise SystemExit("--min-speakers cannot be lower than 30 for this research calibration workflow.")
    if not args.predictions.is_file():
        raise SystemExit(f"Predictions CSV not found: {args.predictions}")

    raw_bytes = args.predictions.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    with args.predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"speaker_id", "actual_age_years", "raw_predicted_age_years", "model_version", "label_source"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"Predictions CSV must include these columns: {', '.join(sorted(required))}.")
        source_rows = list(reader)

    rows: list[dict[str, Any]] = []
    for row in source_rows:
        try:
            actual = float(row["actual_age_years"])
            predicted = float(row["raw_predicted_age_years"])
            if not np.isfinite((actual, predicted)).all() or not 18 <= actual <= 110 or not 0 <= predicted <= 100:
                continue
            if row["label_source"] != "self_reported" or not row["speaker_id"].strip():
                continue
            rows.append({
                "speaker_id": row["speaker_id"].strip(),
                "actual_age_years": actual,
                "raw_predicted_age_years": predicted,
                "baseline_age_group": age_group(predicted),
            })
        except (TypeError, ValueError):
            continue

    speakers = sorted({row["speaker_id"] for row in rows})
    model_versions = {str(row.get("model_version", "")).strip() for row in source_rows}
    model_versions.discard("")
    if len(model_versions) != 1:
        raise SystemExit("Predictions must contain exactly one non-empty model_version.")
    model_version = next(iter(model_versions))
    if model_version != "audeering-wav2vec2-age-gender-6-layer-research":
        raise SystemExit("This calibration workflow is restricted to the named research model version.")
    if len(speakers) < args.min_speakers:
        raise SystemExit(
            f"Need at least {args.min_speakers} distinct pseudonymous speakers; found {len(speakers)}. "
            "Do not lower this threshold to force a result."
        )

    random.Random(args.seed).shuffle(speakers)
    test_count = max(10, int(round(len(speakers) * 0.25)))
    test_speakers = set(speakers[:test_count])
    train_speakers = set(speakers[test_count:])
    if len(test_speakers) < 10 or len(train_speakers) < 20:
        raise SystemExit("Speaker split did not produce at least 20 train and 10 held-out speakers.")
    train = [row for row in rows if row["speaker_id"] in train_speakers]
    test = [row.copy() for row in rows if row["speaker_id"] in test_speakers]
    if len(train) < 30 or len(test) < 10:
        raise SystemExit("Too few records after the speaker-separated split.")

    x = np.asarray([row["raw_predicted_age_years"] for row in train], dtype=np.float64)
    y = np.asarray([row["actual_age_years"] for row in train], dtype=np.float64)
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = (float(value) for value in np.linalg.lstsq(design, y, rcond=None)[0])
    if not np.isfinite((slope, intercept)).all() or not (0.25 <= slope <= 2.0) or not (-40 <= intercept <= 40):
        raise SystemExit(
            "Fitted parameters are outside guarded research bounds. More data or a different model may be required."
        )

    # Match runtime policy: an adult-only calibrator is not applied when the raw
    # model predicts under 18, since this could harm predictions for minors.
    for row in test:
        raw = float(row["raw_predicted_age_years"])
        adjusted = min(100.0, max(18.0, slope * raw + intercept)) if raw >= 18.0 else raw
        row["calibrated_predicted_age_years"] = adjusted
    baseline_mae = mae(test, "raw_predicted_age_years")
    calibrated_mae = mae(test, "calibrated_predicted_age_years")
    baseline_accuracy = group_accuracy(test, "raw_predicted_age_years")
    calibrated_accuracy = group_accuracy(test, "calibrated_predicted_age_years")
    improvement = baseline_mae - calibrated_mae
    eligible = improvement >= 0.5 and calibrated_accuracy >= baseline_accuracy - 0.02

    report = {
        "schema_version": 1,
        "method": "affine_age_calibration",
        "model_version": model_version,
        "slope": round(slope, 8),
        "intercept": round(intercept, 8),
        "train_speaker_count": len(train_speakers),
        "train_record_count": len(train),
        "test_speaker_count": len(test_speakers),
        "test_record_count": len(test),
        "test_split_seed": args.seed,
        "source_predictions_sha256": digest,
        "label_source": "self-reported adult age; not independently verified",
        "baseline_test_mae_years": round(baseline_mae, 4),
        "calibrated_test_mae_years": round(calibrated_mae, 4),
        "held_out_mae_improvement_years": round(improvement, 4),
        "baseline_test_age_group_accuracy": round(baseline_accuracy, 4),
        "calibrated_test_age_group_accuracy": round(calibrated_accuracy, 4),
        "performance_gate_passed": eligible,
        "approved_for_research": False,
        "calibration_scope": "Apply to raw age estimates >= 18 only; other estimates remain uncalibrated.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "This candidate was fitted only on calibration speakers; stated metrics are from held-out speakers.",
            "The held-out group is small for production evidence and labels are self-reported.",
            "This is an experimental affine calibration, not model fine-tuning and not production validation.",
            "A human reviewer must inspect the evidence and manually approve research use before activation.",
            "This research model's weights remain restricted to non-commercial use.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Distinct speakers: {len(speakers)} (train={len(train_speakers)}, held-out={len(test_speakers)})")
    print(f"Baseline held-out MAE: {baseline_mae:.2f} years")
    print(f"Calibrated held-out MAE: {calibrated_mae:.2f} years")
    print(f"Baseline age-group accuracy: {baseline_accuracy:.3f}")
    print(f"Calibrated age-group accuracy: {calibrated_accuracy:.3f}")
    print(f"Performance gate: {'PASS' if eligible else 'FAIL'}")
    print(f"Candidate: {args.output}")
    print("approved_for_research=false; candidate is NOT active and the model is NOT updated automatically.")
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
