from __future__ import annotations

import importlib
import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

AGE_GROUPS = {"Under 13", "13–17", "18–29", "30–44", "45–59", "60+"}
PRESENTATIONS = {"Feminine-coded", "Masculine-coded", "Uncertain"}
RESEARCH_LIMITATIONS = [
    "Research-model outputs have not been validated on a representative VGRS speech set.",
    "Voice-based estimates are not verified chronological age or gender identity.",
    "The optional public audEERING weights are non-commercial only and English-focused; do not use for commercial deployment.",
]


class ClassifierConfigurationError(RuntimeError):
    """The requested classifier is misconfigured or not approved for this environment."""


class ClassifierAdapter(Protocol):
    model_version: str

    def predict(self, waveform: np.ndarray, sample_rate: int) -> Mapping[str, Any]:
        """Return normalized age_group and vocal_presentation labels."""


def _bool_env(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _age_group(age_years: float) -> str:
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


class AudEERINGResearchAdapter:
    """Explicitly opt-in adapter for the non-commercial audEERING 6-layer ONNX model."""

    model_version = "audeering-wav2vec2-age-gender-6-layer-research"

    def __init__(self, model_dir: Path) -> None:
        try:
            import audonnx
        except ImportError as exc:
            raise ClassifierConfigurationError(
                "Research classifier selected but audonnx is not installed. "
                "Install backend/requirements-classifier-research.txt."
            ) from exc
        if not model_dir.is_dir():
            raise ClassifierConfigurationError(
                "VGRS_CLASSIFIER_MODEL_DIR must point to the extracted audEERING ONNX model directory."
            )
        try:
            self._model = audonnx.load(str(model_dir))
        except Exception as exc:
            raise ClassifierConfigurationError(
                "The configured research model could not be loaded. Verify the ONNX model directory."
            ) from exc
        self._age_calibration: dict[str, Any] | None = None
        calibration_path = os.getenv("VGRS_RESEARCH_AGE_CALIBRATION", "").strip()
        if calibration_path:
            self._age_calibration = self._load_age_calibration(Path(calibration_path).expanduser())

    def _load_age_calibration(self, path: Path) -> dict[str, Any]:
        """Load only a manually approved, held-out-tested research calibration."""
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClassifierConfigurationError(
                "Configured research age calibration could not be read as valid JSON."
            ) from exc
        if not isinstance(report, dict) or report.get("approved_for_research") is not True:
            raise ClassifierConfigurationError(
                "Research age calibration is a candidate only. Review it and set approved_for_research=true before enabling."
            )
        if report.get("model_version") != self.model_version:
            raise ClassifierConfigurationError(
                "Research age calibration model_version does not match the loaded classifier."
            )
        try:
            slope = float(report["slope"])
            intercept = float(report["intercept"])
            train_speakers = int(report["train_speaker_count"])
            test_speakers = int(report["test_speaker_count"])
            baseline_mae = float(report["baseline_test_mae_years"])
            calibrated_mae = float(report["calibrated_test_mae_years"])
            baseline_accuracy = float(report["baseline_test_age_group_accuracy"])
            calibrated_accuracy = float(report["calibrated_test_age_group_accuracy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ClassifierConfigurationError(
                "Research age calibration is missing valid held-out evaluation metrics."
            ) from exc
        numeric = (slope, intercept, baseline_mae, calibrated_mae, baseline_accuracy, calibrated_accuracy)
        if not np.isfinite(numeric).all() or not (0.25 <= slope <= 2.0) or not (-40.0 <= intercept <= 40.0):
            raise ClassifierConfigurationError("Research age calibration parameters are outside safe bounds.")
        if train_speakers < 20 or test_speakers < 10:
            raise ClassifierConfigurationError(
                "Research age calibration needs at least 20 calibration speakers and 10 held-out test speakers."
            )
        if baseline_mae - calibrated_mae < 0.5 or calibrated_accuracy < baseline_accuracy - 0.02:
            raise ClassifierConfigurationError(
                "Research age calibration did not meet the held-out improvement gate."
            )
        return {
            **report,
            "slope": slope,
            "intercept": intercept,
            "baseline_test_mae_years": baseline_mae,
            "calibrated_test_mae_years": calibrated_mae,
        }

    def predict(self, waveform: np.ndarray, sample_rate: int) -> Mapping[str, Any]:
        signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            raise ValueError("Classifier input must be mono 16 kHz audio.")
        if signal.size < sample_rate * 2:
            raise ValueError("Classifier needs at least two seconds of audio.")
        outputs = self._model(signal, sample_rate)
        age_values = np.asarray(outputs.get("logits_age"), dtype=np.float64).reshape(-1)
        gender_logits = np.asarray(outputs.get("logits_gender"), dtype=np.float64).reshape(-1)
        if age_values.size != 1 or gender_logits.size != 3:
            raise ValueError("The configured audEERING model returned an unexpected output shape.")
        age_scalar = float(age_values[0])
        if not np.isfinite(age_scalar) or not np.isfinite(gender_logits).all():
            raise ValueError("The configured audEERING model returned non-finite outputs.")
        # The published model emits a normalized age estimate in approximately [0, 1].
        if not 0.0 <= age_scalar <= 1.0:
            raise ValueError("The configured audEERING model age output is outside the documented range.")
        raw_estimated_age_years = age_scalar * 100.0
        estimated_age_years = raw_estimated_age_years
        calibration_applied = False
        # Calibration is fitted on consenting adults only. Avoid applying it to
        # model outputs below 18, where it could worsen predictions for minors.
        if getattr(self, "_age_calibration", None) is not None and raw_estimated_age_years >= 18.0:
            estimated_age_years = float(np.clip(
                self._age_calibration["slope"] * raw_estimated_age_years
                + self._age_calibration["intercept"],
                18.0,
                100.0,
            ))
            calibration_applied = True
        # Published order: female, male, child. A child prediction is not turned into a
        # female/male label because those classes are not interchangeable.
        class_index = int(np.argmax(gender_logits))
        presentation: str | None
        if class_index == 0:
            presentation = "Feminine-coded"
        elif class_index == 1:
            presentation = "Masculine-coded"
        else:
            presentation = None
        return {
            "age_group": _age_group(estimated_age_years),
            "estimated_age_years": round(estimated_age_years, 2),
            "calibration_applied": calibration_applied,
            "vocal_presentation": presentation,
            "model_version": self.model_version,
            "status": "experimental",
            "limitations": RESEARCH_LIMITATIONS,
            "vocal_presentation_abstained": class_index == 2,
        }


def _load_plugin(spec: str) -> ClassifierAdapter:
    if ":" not in spec:
        raise ClassifierConfigurationError(
            "VGRS_CLASSIFIER_PLUGIN must use 'python.module:factory_function' syntax."
        )
    module_name, factory_name = spec.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), factory_name)
        adapter = factory()
    except Exception as exc:
        raise ClassifierConfigurationError(
            "The configured production classifier plugin could not be initialized."
        ) from exc
    if not callable(getattr(adapter, "predict", None)):
        raise ClassifierConfigurationError("Classifier plugin must expose predict(waveform, sample_rate).")
    if not str(getattr(adapter, "model_version", "")).strip():
        raise ClassifierConfigurationError("Classifier plugin must declare a non-empty model_version.")
    return adapter


def _read_validation_report() -> dict[str, Any] | None:
    report_path = os.getenv("VGRS_CLASSIFIER_VALIDATION_REPORT", "").strip()
    if not report_path:
        return None
    try:
        data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    required = (
        "production_approved", "model_version", "dataset_version", "speaker_count",
        "macro_f1", "balanced_accuracy", "per_class_metrics", "subgroup_metrics",
        "calibration", "abstention_coverage", "evaluation_date", "label_definition",
        "consent_and_dataset_rights_reviewed",
    )
    if any(key not in data for key in required):
        return None
    if data.get("production_approved") is not True or data.get("consent_and_dataset_rights_reviewed") is not True:
        return None
    if not isinstance(data.get("model_version"), str) or not data["model_version"].strip():
        return None
    if not isinstance(data.get("dataset_version"), str) or not data["dataset_version"].strip():
        return None
    if not isinstance(data.get("speaker_count"), int) or data["speaker_count"] < 500:
        return None
    for key in ("macro_f1", "balanced_accuracy", "abstention_coverage"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            return None
    if not isinstance(data.get("per_class_metrics"), dict) or not data["per_class_metrics"]:
        return None
    if not isinstance(data.get("subgroup_metrics"), dict) or not data["subgroup_metrics"]:
        return None
    if not isinstance(data.get("calibration"), dict) or not data["calibration"]:
        return None
    if not isinstance(data.get("evaluation_date"), str) or not data["evaluation_date"].strip():
        return None
    try:
        evaluated_on = date.fromisoformat(data["evaluation_date"])
    except ValueError:
        return None
    today = date.today()
    if evaluated_on > today or evaluated_on < today - timedelta(days=365):
        return None
    if not isinstance(data.get("label_definition"), str) or not data["label_definition"].strip():
        return None
    return data


class ClassifierRuntime:
    def __init__(self) -> None:
        self.adapter: ClassifierAdapter | None = None
        self.status = "not_configured"
        self.backend = os.getenv("VGRS_CLASSIFIER_BACKEND", "none").strip().lower()
        self.environment = os.getenv("VGRS_ENV", "development").strip().lower()
        self.validation_report: dict[str, Any] | None = None

    def configure(self) -> None:
        # Startup/configuration may be retried in tests or worker lifecycle events.
        self.adapter = None
        self.validation_report = None
        self.status = "not_configured"
        if self.backend in {"", "none", "disabled"}:
            if self.environment == "production":
                raise ClassifierConfigurationError(
                    "Production startup blocked: a licensed, validated classifier is not configured."
                )
            self.status = "not_configured"
            return

        if self.backend == "audeering-research":
            if self.environment == "production":
                raise ClassifierConfigurationError(
                    "The public audEERING research weights are non-commercial only and cannot be enabled in production."
                )
            if not _bool_env("VGRS_ENABLE_NONCOMMERCIAL_CLASSIFIER"):
                raise ClassifierConfigurationError(
                    "Set VGRS_ENABLE_NONCOMMERCIAL_CLASSIFIER=true only if your use complies with the model's non-commercial license."
                )
            model_dir = Path(os.getenv("VGRS_CLASSIFIER_MODEL_DIR", "").strip()).expanduser()
            self.adapter = AudEERINGResearchAdapter(model_dir)
            self.status = "experimental"
            return

        if self.backend == "custom-plugin":
            plugin_spec = os.getenv("VGRS_CLASSIFIER_PLUGIN", "").strip()
            self.adapter = _load_plugin(plugin_spec)
            self.validation_report = _read_validation_report()
            license_approved = _bool_env("VGRS_CLASSIFIER_LICENSE_APPROVED")
            production_approved = bool(
                self.validation_report
                and self.validation_report.get("production_approved") is True
                and str(self.validation_report.get("model_version")) == str(self.adapter.model_version)
            )
            if self.environment == "production" and not license_approved:
                raise ClassifierConfigurationError(
                    "Production startup blocked: VGRS_CLASSIFIER_LICENSE_APPROVED=true is required after legal review."
                )
            if self.environment == "production" and not production_approved:
                raise ClassifierConfigurationError(
                    "Production startup blocked: a matching, production-approved classifier validation report is required."
                )
            self.status = "available" if production_approved and license_approved else "experimental"
            return

        raise ClassifierConfigurationError(
            "Unsupported VGRS_CLASSIFIER_BACKEND. Choose none, audeering-research, or custom-plugin."
        )

    @property
    def production_ready(self) -> bool:
        return self.status == "available" and self.validation_report is not None

    def metadata(self) -> dict[str, Any]:
        if self.adapter is None:
            return {
                "id": "age-presentation-classifier",
                "version": None,
                "kind": "classification",
                "status": "not_configured",
                "license": None,
                "accuracy_established": False,
                "production_approved": False,
                "limitations": ["No licensed classifier is loaded; predictions are not fabricated."],
            }
        if self.backend == "audeering-research":
            return {
                "id": "audeering-wav2vec2-age-gender-6-layer",
                "version": self.adapter.model_version,
                "kind": "classification",
                "status": "experimental",
                "license": "CC BY-NC-SA 4.0 — non-commercial only",
                "accuracy_established": False,
                "production_approved": False,
                "age_calibration": (
                    {
                        "status": "active",
                        "method": "affine",
                        "held_out_mae_improvement_years": round(
                            self.adapter._age_calibration["baseline_test_mae_years"]
                            - self.adapter._age_calibration["calibrated_test_mae_years"], 3
                        ),
                        "scope": "adult-range outputs only",
                    }
                    if getattr(self.adapter, "_age_calibration", None)
                    else {"status": "not_configured"}
                ),
                "limitations": RESEARCH_LIMITATIONS,
            }
        report = self.validation_report or {}
        return {
            "id": "vgrs-custom-classifier",
            "version": self.adapter.model_version,
            "kind": "classification",
            "status": self.status,
            "license": "Reviewed by deployer",
            "accuracy_established": self.production_ready,
            "production_approved": self.production_ready,
            "limitations": [
                "Model accuracy claims are valid only for the population and protocol in the validation report.",
                "Predictions estimate perceived vocal presentation and perceived age group, not gender identity or verified age.",
            ],
        }

    def predict(self, waveform: np.ndarray, sample_rate: int) -> dict[str, Any] | None:
        if self.adapter is None:
            return None
        raw = self.adapter.predict(waveform, sample_rate)
        if not isinstance(raw, Mapping):
            raise ValueError("Classifier adapter returned an invalid result.")
        age_group = raw.get("age_group")
        presentation = raw.get("vocal_presentation")
        if age_group not in AGE_GROUPS and age_group is not None:
            raise ValueError("Classifier returned an unsupported age-group label.")
        if presentation not in PRESENTATIONS and presentation is not None:
            raise ValueError("Classifier returned an unsupported vocal-presentation label.")
        expected_version = self.adapter.model_version
        if raw.get("model_version") not in (None, expected_version):
            raise ValueError("Classifier returned a model version inconsistent with its loaded model.")
        # The adapter is never allowed to mark itself production-approved. Only the
        # host runtime's license + matching validation gate can grant "available".
        status = "available" if self.production_ready else "experimental"
        limitations = raw.get("limitations", [])
        note = "; ".join(str(item) for item in limitations) if limitations else (
            "This is an estimate of perceived vocal presentation and perceived age group, not gender identity or verified chronological age."
        )
        if raw.get("vocal_presentation_abstained"):
            note += " The model selected a child class, so an adult-coded vocal-presentation label was withheld."
        common = {
            "model_version": expected_version,
            "scores": None,
        }
        if isinstance(raw.get("estimated_age_years"), (int, float)) and np.isfinite(raw["estimated_age_years"]):
            common["estimated_age_years"] = float(raw["estimated_age_years"])
            common["calibration_applied"] = bool(raw.get("calibration_applied", False))
        return {
            "age_group": {
                **common, "status": "abstained" if age_group is None else status, "label": age_group,
                "note": note if age_group is not None else "The classifier abstained on age-group estimation.",
            },
            "vocal_presentation": {
                **common, "status": "abstained" if presentation is None else status,
                "label": presentation,
                "note": note if presentation is not None else (
                    "The classifier did not return a supported vocal-presentation label; no label was inferred."
                ),
            },
        }


classifier_runtime = ClassifierRuntime()
