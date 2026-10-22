import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "ml" / "evaluation" / "split_by_speaker.py"
EVALUATE = ROOT / "ml" / "evaluation" / "evaluate_predictions.py"


def test_split_tool_keeps_speakers_and_declared_duplicates_together(tmp_path):
    source = tmp_path / "manifest.csv"
    output = tmp_path / "split.csv"
    columns = ["recording_id", "speaker_id", "label", "duplicate_group_id"]
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for speaker_number in range(30):
            label = "masculine" if speaker_number % 2 == 0 else "feminine"
            for take in range(2):
                duplicate = "same-content-group" if (speaker_number == 0 and take == 0) or (speaker_number == 1 and take == 1) else ""
                writer.writerow({"recording_id": f"r-{speaker_number}-{take}", "speaker_id": f"s-{speaker_number}",
                                 "label": label, "duplicate_group_id": duplicate})

    completed = subprocess.run(
        [sys.executable, str(SPLIT), "--manifest", str(source), "--output", str(output), "--seed", "7"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    split = pd.read_csv(output, dtype=str)
    assert set(split["split"]) == {"train", "validation", "test"}
    assert split.groupby("speaker_id")["split"].nunique().max() == 1
    linked = split[split["duplicate_group_id"] == "same-content-group"]
    assert linked["split"].nunique() == 1
    assert "Leakage assertion: PASS" in completed.stdout


def test_evaluation_cli_reports_abstention_coverage_and_metrics(tmp_path):
    source = tmp_path / "predictions.csv"
    output = tmp_path / "metrics.json"
    pd.DataFrame([
        {"speaker_id": "s1", "true_label": "feminine", "predicted_label": "feminine", "prob_feminine": 0.9, "prob_masculine": 0.1, "accent": "A"},
        {"speaker_id": "s2", "true_label": "masculine", "predicted_label": "uncertain", "prob_feminine": 0.4, "prob_masculine": 0.6, "accent": "B"},
        {"speaker_id": "s3", "true_label": "masculine", "predicted_label": "masculine", "prob_feminine": 0.1, "prob_masculine": 0.9, "accent": "B"},
    ]).to_csv(source, index=False)

    completed = subprocess.run(
        [sys.executable, str(EVALUATE), "--predictions", str(source), "--output", str(output),
         "--speaker-col", "speaker_id", "--subgroup-col", "accent", "--bootstrap", "50"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    metrics = report["primary_metrics"]
    assert metrics["n"] == 3
    assert metrics["n_abstained"] == 1
    assert abs(metrics["coverage"] - 2 / 3) < 1e-8
    assert metrics["macro_f1_on_covered"] == 1.0
    assert report["probability_metrics"]["available"] is True
    assert set(report["subgroups"]) == {"A", "B"}
    assert report["speaker_bootstrap"]["unit"] == "speaker cluster"
