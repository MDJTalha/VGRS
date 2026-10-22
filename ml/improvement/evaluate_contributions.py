from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.audio_pipeline import AudioValidationError, analyze_audio  # noqa: E402

MODEL_VERSION = "audeering-wav2vec2-age-gender-6-layer-research"


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


def _metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    if not rows:
        return {"record_count": 0, "mae_years": None, "median_absolute_error_years": None,
                "within_5_years_rate": None, "age_group_accuracy": None, "per_actual_age_group": {}}
    errors = [abs(float(row[prediction_key]) - float(row["actual_age_years"])) for row in rows]
    accurate_groups = [
        age_group(float(row[prediction_key])) == age_group(float(row["actual_age_years"]))
        for row in rows
    ]
    per_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_group[age_group(float(row["actual_age_years"]))].append(row)
    return {
        "record_count": len(rows),
        "mae_years": round(mean(errors), 3),
        "median_absolute_error_years": round(median(errors), 3),
        "within_5_years_rate": round(sum(error <= 5 for error in errors) / len(errors), 4),
        "age_group_accuracy": round(sum(accurate_groups) / len(rows), 4),
        "per_actual_age_group": {
            group: {
                "record_count": len(group_rows),
                "mae_years": round(mean(abs(float(row[prediction_key]) - float(row["actual_age_years"])) for row in group_rows), 3),
                "age_group_accuracy": round(
                    mean(age_group(float(row[prediction_key])) == group for row in group_rows), 4
                ),
            }
            for group, group_rows in sorted(per_group.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the non-commercial research age model on explicitly contributed, self-reported adult samples."
    )
    parser.add_argument("--contributions-dir", type=Path, default=ROOT / "data" / "research-contributions")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "audeering-age-gender-6-layer")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "research-improvement" / "predictions.csv")
    args = parser.parse_args()

    try:
        import audonnx
    except ImportError as exc:
        raise SystemExit(
            "audonnx is missing. Install backend\\requirements-classifier-research.txt in backend\\.venv."
        ) from exc

    if not args.contributions_dir.is_dir():
        raise SystemExit(
            f"No contribution directory exists at {args.contributions_dir}. "
            "Enable research contributions only after reading docs/continuous-improvement.md."
        )
    if not args.model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_dir}")

    model = audonnx.load(str(args.model_dir))
    metadata_files = sorted(args.contributions_dir.glob("*/metadata.json"))
    predictions: list[dict[str, Any]] = []
    metadata_hash = hashlib.sha256()

    for metadata_path in metadata_files:
        try:
            metadata_bytes = metadata_path.read_bytes()
            metadata_hash.update(metadata_bytes)
            meta = json.loads(metadata_bytes.decode("utf-8"))
            if meta.get("consent_to_research") is not True:
                continue
            if meta.get("label_source") != "self_reported":
                continue
            actual_age = float(meta["actual_age_years"])
            if not 18 <= actual_age <= 110:
                continue
            contribution_id = str(meta["contribution_id"])
            speaker_id = str(meta.get("speaker_id") or contribution_id)
            audio_name = str(meta["audio_file"])
            if Path(audio_name).name != audio_name or not audio_name.startswith("recording."):
                raise ValueError("invalid audio file name")
            audio_path = metadata_path.parent / audio_name
            if not audio_path.is_file():
                raise ValueError("contributed audio is missing")
            payload = audio_path.read_bytes()
            analysis = analyze_audio(payload, audio_path.name)
            outputs = model(analysis.waveform.astype(np.float32, copy=False), 16000)
            raw_age = np.asarray(outputs.get("logits_age"), dtype=np.float64).reshape(-1)
            if raw_age.size != 1 or not np.isfinite(raw_age).all() or not 0 <= float(raw_age[0]) <= 1:
                raise ValueError("unexpected age output")
            predicted = float(raw_age[0]) * 100.0
            predictions.append({
                "contribution_id": contribution_id,
                "speaker_id": speaker_id,
                "actual_age_years": round(actual_age, 2),
                "actual_age_group": age_group(actual_age),
                "raw_predicted_age_years": round(predicted, 4),
                "raw_predicted_age_group": age_group(predicted),
                "absolute_error_years": round(abs(predicted - actual_age), 4),
                "model_version": MODEL_VERSION,
                "label_source": "self_reported",
                "audio_quality_status": str(meta.get("audio_quality_status", "unknown")),
            })
        except (OSError, KeyError, TypeError, ValueError, AudioValidationError):
            # Do not log audio paths, speaker codes, ages or contributor identifiers.
            print("Skipped one contribution: invalid metadata, audio or model output.", file=sys.stderr)

    if not predictions:
        raise SystemExit("No valid, consented adult research contributions were available for evaluation.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "contribution_id", "speaker_id", "actual_age_years", "actual_age_group",
        "raw_predicted_age_years", "raw_predicted_age_group", "absolute_error_years",
        "model_version", "label_source", "audio_quality_status",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    report = {
        "report_type": "research_age_model_evaluation",
        "model_version": MODEL_VERSION,
        "evaluated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "contribution_metadata_sha256": metadata_hash.hexdigest(),
        "unique_speakers": len({row["speaker_id"] for row in predictions}),
        "label_source": "self-reported adult age; not independently verified",
        "metrics": _metrics(predictions, "raw_predicted_age_years"),
        "limitations": [
            "This is a volunteer/contributor sample, not necessarily representative of the launch population.",
            "Self-reported ages may be inaccurate. Check label quality and rights before model use.",
            "If a speaker submitted multiple recordings, the same pseudonymous participant code must be reused.",
            "These metrics do not establish clinical, identity, chronological-age verification or production suitability.",
            "This script evaluates the existing model only; it does not train, calibrate or promote a model.",
        ],
        "predictions_csv": str(args.output),
        "production_approved": False,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    metrics = report["metrics"]
    print(f"Evaluated records: {metrics['record_count']}")
    print(f"Distinct speakers: {report['unique_speakers']}")
    print(f"Mean absolute age error: {metrics['mae_years']} years")
    print(f"Age-group accuracy: {metrics['age_group_accuracy']}")
    print(f"Prediction CSV: {args.output}")
    print(f"Summary report: {summary_path}")
    print("This evaluation does not approve the model for production.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
