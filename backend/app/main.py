from __future__ import annotations
import os
import uuid
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .audio_pipeline import ALLOWED_EXTENSIONS, AudioValidationError, analyze_audio

APP_VERSION = "0.1.0"
MAX_UPLOAD_BYTES = int(float(os.getenv("VGRS_MAX_UPLOAD_MB", "25")) * 1024 * 1024)
MAX_DURATION = float(os.getenv("VGRS_MAX_DURATION_SECONDS", "120"))
MIN_DURATION = float(os.getenv("VGRS_MIN_DURATION_SECONDS", "2"))
MODE = os.getenv("VGRS_MODE", "analysis-only").strip().lower()
ENVIRONMENT = os.getenv("VGRS_ENV", "development").strip().lower()
ORIGINS = [v.strip() for v in os.getenv("VGRS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if v.strip()]

app = FastAPI(title="VGRS — Voice Analysis API", version=APP_VERSION,
              description="Local-first acoustic analysis. Voice alone cannot establish gender identity or chronological age.")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Accept"])

class Prediction(BaseModel):
    status: str
    label: str | None = None
    scores: dict[str, float] | None = None
    model_version: str | None = None
    note: str

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    filename: str
    audio_quality: dict[str, Any]
    vocal_presentation: Prediction
    age_group: Prediction
    acoustic_features: dict[str, Any]
    model_info: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)

def _analyze(payload: bytes, filename: str) -> AnalysisResponse:
    try:
        result = analyze_audio(payload, filename, max_duration=MAX_DURATION, min_duration=MIN_DURATION)
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_audio", "message": str(exc)}) from exc
    unavailable = "No compatible real classification model is enabled. VGRS does not guess demographic attributes from pitch or substitute synthetic predictions."
    return AnalysisResponse(
        analysis_id=str(uuid.uuid4()), status="completed", filename=filename,
        audio_quality=result.quality,
        vocal_presentation=Prediction(status="model_unavailable", note=unavailable),
        age_group=Prediction(status="model_unavailable", note=unavailable),
        acoustic_features=result.features,
        model_info={"feature_pipeline": "vgrs-acoustic-baseline", "feature_pipeline_version": APP_VERSION,
                    "classification_model": None, "mode": MODE, "model_enabled": False, "accuracy_established": False},
        limitations=[
            "Acoustic measurements do not establish gender identity or verified chronological age.",
            "Classification is unavailable until a suitable licensed model is evaluated and enabled.",
            "No accuracy has been established on a representative, speaker-independent VGRS test set.",
            "Accent, language, recording equipment, vocal training, health and noise may affect measurements.",
        ])

@app.get("/")
def root() -> dict[str, str]:
    return {"name": "VGRS Voice Analysis API", "version": APP_VERSION, "docs": "/docs"}

@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": APP_VERSION, "mode": MODE,
            "environment": ENVIRONMENT, "classification_ready": False}

@app.get("/api/v1/models")
def models() -> dict[str, Any]:
    return {"models": [
        {"id": "vgrs-acoustic-baseline", "version": APP_VERSION, "kind": "acoustic_features",
         "status": "available", "accuracy_established": False,
         "outputs": ["f0_median_hz", "f0_p10_hz", "f0_p90_hz", "spectral_centroid_hz",
                     "spectral_rolloff_85_hz", "zero_crossing_rate", "active_audio_fraction"],
         "limitations": ["Feature extraction only; no age or vocal-presentation prediction."]},
        {"id": "age-presentation-classifier", "version": None, "kind": "classification",
         "status": "not_configured", "license": None, "accuracy_established": False,
         "limitations": ["No classification model loaded; predictions are not fabricated."]}]}

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...), consent: bool = Form(...)) -> AnalysisResponse:
    if not consent:
        raise HTTPException(status_code=403, detail={"code": "consent_required", "message": "Consent is required before audio analysis."})
    filename = (file.filename or "upload").replace("\\", "/").split("/")[-1] or "upload"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail={"code": "unsupported_format", "message": "Unsupported audio extension."})
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail={"code": "file_too_large", "message": f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB."})
    return _analyze(payload, filename)

@app.post("/api/v1/analyze/batch")
async def analyze_batch(files: list[UploadFile] = File(...), consent: bool = Form(...)) -> dict[str, Any]:
    if not consent:
        raise HTTPException(status_code=403, detail={"code": "consent_required", "message": "Consent is required before batch audio analysis."})
    if not files:
        raise HTTPException(status_code=422, detail={"code": "empty_batch", "message": "Select at least one audio file."})
    if len(files) > 20:
        raise HTTPException(status_code=413, detail={"code": "batch_too_large", "message": "Maximum 20 files per batch."})
    rows = []
    for upload in files:
        filename = (upload.filename or "upload").replace("\\", "/").split("/")[-1] or "upload"
        try:
            payload = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(payload) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail={"code": "file_too_large", "message": "File exceeds upload limit."})
            rows.append({"filename": filename, "status": "completed", "result": _analyze(payload, filename).model_dump()})
        except HTTPException as exc:
            rows.append({"filename": filename, "status": "failed", "error": exc.detail})
        finally:
            await upload.close()
    return {"count": len(rows), "completed": sum(x["status"] == "completed" for x in rows),
            "failed": sum(x["status"] == "failed" for x in rows), "results": rows}
