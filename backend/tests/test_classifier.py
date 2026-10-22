import json
import sys
from datetime import date
from types import ModuleType

import numpy as np
import pytest

from app.classifier import (
    AudEERINGResearchAdapter,
    ClassifierConfigurationError,
    ClassifierRuntime,
)


class FakeResearchModel:
    def __init__(self, age=0.32, gender=(0.2, 1.3, -0.4)):
        self.age = age
        self.gender = gender

    def __call__(self, waveform, sample_rate):
        assert sample_rate == 16000
        return {
            "logits_age": np.asarray([[self.age]], dtype=np.float32),
            "logits_gender": np.asarray([self.gender], dtype=np.float32),
        }


def test_research_adapter_maps_documented_outputs_without_confidence_scores():
    adapter = AudEERINGResearchAdapter.__new__(AudEERINGResearchAdapter)
    adapter._model = FakeResearchModel(age=0.32, gender=(0.2, 1.3, -0.4))
    result = adapter.predict(np.zeros(16000 * 3, dtype=np.float32), 16000)

    assert result["age_group"] == "30–44"
    assert result["vocal_presentation"] == "Masculine-coded"
    assert result["status"] == "experimental"
    assert "not been validated" in result["limitations"][0]
    assert "confidence" not in result


def test_research_adapter_abstains_on_child_class_for_vocal_presentation():
    adapter = AudEERINGResearchAdapter.__new__(AudEERINGResearchAdapter)
    adapter._model = FakeResearchModel(age=0.1, gender=(0.1, 0.2, 2.0))
    result = adapter.predict(np.zeros(16000 * 3, dtype=np.float32), 16000)

    assert result["age_group"] == "Under 13"
    assert result["vocal_presentation"] is None
    assert result["vocal_presentation_abstained"] is True


def test_research_adapter_rejects_nonfinite_model_outputs():
    adapter = AudEERINGResearchAdapter.__new__(AudEERINGResearchAdapter)
    adapter._model = FakeResearchModel(age=float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        adapter.predict(np.zeros(16000 * 3, dtype=np.float32), 16000)


def test_production_refuses_to_start_without_a_classifier(monkeypatch):
    runtime = ClassifierRuntime()
    runtime.environment = "production"
    runtime.backend = "none"

    with pytest.raises(ClassifierConfigurationError, match="classifier is not configured"):
        runtime.configure()


def test_custom_classifier_requires_matching_validated_license_report(tmp_path, monkeypatch):
    class FakeCustomClassifier:
        model_version = "demo-2026-10"

        def predict(self, waveform, sample_rate):
            return {
                "age_group": "18–29",
                "vocal_presentation": "Feminine-coded",
                "model_version": self.model_version,
                "status": "available",
            }

    module = ModuleType("vgrs_test_classifier_plugin")
    module.create_classifier = lambda: FakeCustomClassifier()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    report = tmp_path / "validation.json"
    report.write_text(json.dumps({
        "production_approved": True,
        "consent_and_dataset_rights_reviewed": True,
        "model_version": "demo-2026-10",
        "dataset_version": "consented-speaker-heldout-v1",
        "speaker_count": 500,
        "macro_f1": 0.82,
        "balanced_accuracy": 0.84,
        "per_class_metrics": {"Feminine-coded": {"precision": 0.83, "recall": 0.84}},
        "subgroup_metrics": {"en-US": {"macro_f1": 0.80}},
        "calibration": {"method": "temperature-scaling", "expected_calibration_error": 0.04},
        "abstention_coverage": 0.91,
        "evaluation_date": date.today().isoformat(),
        "label_definition": "Perceived vocal presentation and perceived age group.",
    }), encoding="utf-8")
    monkeypatch.setenv("VGRS_CLASSIFIER_BACKEND", "custom-plugin")
    monkeypatch.setenv("VGRS_CLASSIFIER_PLUGIN", "vgrs_test_classifier_plugin:create_classifier")
    monkeypatch.setenv("VGRS_CLASSIFIER_LICENSE_APPROVED", "true")
    monkeypatch.setenv("VGRS_CLASSIFIER_VALIDATION_REPORT", str(report))

    runtime = ClassifierRuntime()
    runtime.environment = "production"
    runtime.backend = "custom-plugin"
    runtime.configure()

    result = runtime.predict(np.zeros(16000 * 3, dtype=np.float32), 16000)
    assert runtime.production_ready is True
    assert result["age_group"]["label"] == "18–29"
    assert result["vocal_presentation"]["label"] == "Feminine-coded"
    assert result["age_group"]["scores"] is None
    assert runtime.metadata()["production_approved"] is True


def test_research_classifier_is_blocked_in_production(monkeypatch):
    monkeypatch.setenv("VGRS_ENABLE_NONCOMMERCIAL_CLASSIFIER", "true")
    runtime = ClassifierRuntime()
    runtime.environment = "production"
    runtime.backend = "audeering-research"

    with pytest.raises(ClassifierConfigurationError, match="cannot be enabled in production"):
        runtime.configure()



def test_validation_report_rejects_boolean_metrics_and_expired_evidence(tmp_path, monkeypatch):
    from app.classifier import _read_validation_report

    base = {
        "production_approved": True,
        "consent_and_dataset_rights_reviewed": True,
        "model_version": "model-v1",
        "dataset_version": "dataset-v1",
        "speaker_count": 500,
        "macro_f1": 0.8,
        "balanced_accuracy": 0.8,
        "per_class_metrics": {"class-a": {"precision": 0.8, "recall": 0.8}},
        "subgroup_metrics": {"language-a": {"macro_f1": 0.75}},
        "calibration": {"method": "temperature-scaling"},
        "abstention_coverage": 0.9,
        "evaluation_date": date.today().isoformat(),
        "label_definition": "Perceived presentation and perceived age.",
    }
    path = tmp_path / "validation.json"
    monkeypatch.setenv("VGRS_CLASSIFIER_VALIDATION_REPORT", str(path))

    path.write_text(json.dumps({**base, "macro_f1": True}), encoding="utf-8")
    assert _read_validation_report() is None

    path.write_text(json.dumps({**base, "evaluation_date": "2000-01-01"}), encoding="utf-8")
    assert _read_validation_report() is None


def _valid_age_calibration(model_version):
    return {
        "schema_version": 1,
        "method": "affine_age_calibration",
        "performance_gate_passed": True,
        "approved_for_research": True,
        "model_version": model_version,
        "slope": 1.1,
        "intercept": 15.0,
        "train_speaker_count": 50,
        "test_speaker_count": 20,
        "baseline_test_mae_years": 10.0,
        "calibrated_test_mae_years": 8.0,
        "baseline_test_age_group_accuracy": 0.6,
        "calibrated_test_age_group_accuracy": 0.6,
    }


def test_research_age_calibration_only_loads_approved_matching_evaluated_candidate(tmp_path):
    from app.classifier import AudEERINGResearchAdapter, ClassifierConfigurationError

    adapter = AudEERINGResearchAdapter.__new__(AudEERINGResearchAdapter)
    adapter.model_version = "audeering-wav2vec2-age-gender-6-layer-research"
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({**_valid_age_calibration(adapter.model_version), "approved_for_research": False}), encoding="utf-8")
    with pytest.raises(ClassifierConfigurationError, match="set approved_for_research=true"):
        adapter._load_age_calibration(candidate)

    candidate.write_text(json.dumps({**_valid_age_calibration("wrong-model")}), encoding="utf-8")
    with pytest.raises(ClassifierConfigurationError, match="does not match"):
        adapter._load_age_calibration(candidate)

    candidate.write_text(json.dumps(_valid_age_calibration(adapter.model_version)), encoding="utf-8")
    loaded = adapter._load_age_calibration(candidate)
    assert loaded["slope"] == 1.1
    assert loaded["test_speaker_count"] == 20


def test_research_age_calibration_changes_adult_age_band_but_never_claims_confidence():
    adapter = AudEERINGResearchAdapter.__new__(AudEERINGResearchAdapter)
    adapter._model = FakeResearchModel(age=0.32, gender=(0.2, 1.3, -0.4))
    adapter._age_calibration = _valid_age_calibration(adapter.model_version)
    result = adapter.predict(np.zeros(16000 * 3, dtype=np.float32), 16000)

    assert result["age_group"] == "45–59"
    assert result["estimated_age_years"] == 50.2
    assert result["calibration_applied"] is True
    assert "confidence" not in result


def test_research_adult_calibration_is_not_applied_to_under_18_raw_estimate():
    adapter = AudEERINGResearchAdapter.__new__(AudEERINGResearchAdapter)
    adapter._model = FakeResearchModel(age=0.15, gender=(0.2, 1.3, -0.4))
    adapter._age_calibration = _valid_age_calibration(adapter.model_version)
    result = adapter.predict(np.zeros(16000 * 3, dtype=np.float32), 16000)

    assert result["age_group"] == "13–17"
    assert result["estimated_age_years"] == 15.0
    assert result["calibration_applied"] is False
