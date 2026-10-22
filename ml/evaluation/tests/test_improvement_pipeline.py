from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from pathlib import Path


def test_age_calibration_pipeline_splits_by_speaker_and_requires_review(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = root / "ml" / "improvement" / "fit_age_calibration.py"
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "candidate.json"
    fields = ["speaker_id", "actual_age_years", "raw_predicted_age_years", "model_version", "label_source"]
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(48):
            actual = 25 + ((index * 11) % 65)
            predicted = (actual - 6) / 1.12 + (((index % 5) - 2) * 0.2)
            writer.writerow({
                "speaker_id": f"speaker-{index:03d}",
                "actual_age_years": actual,
                "raw_predicted_age_years": round(predicted, 4),
                "model_version": "audeering-wav2vec2-age-gender-6-layer-research",
                "label_source": "self_reported",
            })

    completed = subprocess.run(
        [sys.executable, str(script), "--predictions", str(predictions), "--output", str(output), "--seed", "42"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert output.is_file(), completed.stdout + completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["train_speaker_count"] >= 20
    assert report["test_speaker_count"] >= 10
    assert report["train_speaker_count"] + report["test_speaker_count"] == 48
    assert report["approved_for_research"] is False
    assert report["source_predictions_sha256"]
    assert completed.returncode in (0, 2), completed.stdout + completed.stderr
    # Data that clearly improves should pass its held-out acceptance gate.
    assert report["performance_gate_passed"] is True, completed.stdout + completed.stderr


def test_age_calibration_rejects_mixed_model_versions(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = root / "ml" / "improvement" / "fit_age_calibration.py"
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "candidate.json"
    fields = ["speaker_id", "actual_age_years", "raw_predicted_age_years", "model_version", "label_source"]
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(31):
            writer.writerow({
                "speaker_id": f"speaker-{index:03d}",
                "actual_age_years": 30 + index,
                "raw_predicted_age_years": 20 + index,
                "model_version": "model-A" if index < 30 else "model-B",
                "label_source": "self_reported",
            })
    completed = subprocess.run(
        [sys.executable, str(script), "--predictions", str(predictions), "--output", str(output)],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0
    assert "exactly one non-empty model_version" in completed.stderr
