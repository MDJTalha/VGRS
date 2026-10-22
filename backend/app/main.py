from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import ipaddress
import logging
import os
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .audio_pipeline import ALLOWED_EXTENSIONS, AudioValidationError, analyze_audio
from .classifier import ClassifierConfigurationError, classifier_runtime

APP_VERSION = "0.2.0"
MAX_UPLOAD_BYTES = int(float(os.getenv("VGRS_MAX_UPLOAD_MB", "25")) * 1024 * 1024)
MAX_BATCH_BYTES = int(float(os.getenv("VGRS_MAX_BATCH_MB", "100")) * 1024 * 1024)
MAX_DURATION = float(os.getenv("VGRS_MAX_DURATION_SECONDS", "120"))
MIN_DURATION = float(os.getenv("VGRS_MIN_DURATION_SECONDS", "2"))
MAX_CONCURRENT_ANALYSES = max(1, int(os.getenv("VGRS_MAX_CONCURRENT_ANALYSES", "2")))
MODE = os.getenv("VGRS_MODE", "analysis-only").strip().lower()
ENVIRONMENT = os.getenv("VGRS_ENV", "development").strip().lower()
RAW_ORIGINS = os.getenv("VGRS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
ORIGINS = [v.strip() for v in RAW_ORIGINS.split(",") if v.strip()]
TRUSTED_HOSTS = [v.strip() for v in os.getenv("VGRS_TRUSTED_HOSTS", "").split(",") if v.strip()]
TRUSTED_PROXY_IPS = [v.strip() for v in os.getenv("VGRS_TRUSTED_PROXY_IPS", "").split(",") if v.strip()]
try:
    TRUSTED_PROXY_NETWORKS = [ipaddress.ip_network(v, strict=False) for v in TRUSTED_PROXY_IPS]
except ValueError as exc:
    raise RuntimeError("VGRS_TRUSTED_PROXY_IPS must contain valid IP addresses or CIDR ranges.") from exc
PROXY_SHARED_SECRET = os.getenv("VGRS_PROXY_SHARED_SECRET", "")
BODY_OVERHEAD_BYTES = 1024 * 1024
ANALYSIS_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)
logger = logging.getLogger("vgrs.api")

# Training/research collection is disabled by default and deliberately unavailable
# in production. Ordinary analysis never persists uploaded audio.
RESEARCH_DATA_DIR = Path(
    os.getenv(
        "VGRS_RESEARCH_DATA_DIR",
        str(Path(__file__).resolve().parents[2] / "data" / "research-contributions"),
    )
).expanduser()


def _research_contributions_enabled() -> bool:
    return (
        ENVIRONMENT != "production"
        and os.getenv("VGRS_ENABLE_RESEARCH_CONTRIBUTIONS", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def _age_band(age_years: float) -> str:
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


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    filename: str
    audio_quality: dict[str, Any]
    vocal_presentation: dict[str, Any]
    age_group: dict[str, Any]
    acoustic_features: dict[str, Any]
    model_info: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)


class RequestBodyTooLarge(Exception):
    pass


def _apply_rejection_headers(response: JSONResponse, request_id: str, path: str) -> None:
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=()"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    if path.startswith("/api/v1/"):
        response.headers["Cache-Control"] = "no-store"
    if ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"


class RequestGuardMiddleware:
    """Request ID, bounded request bodies and non-sensitive security headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in {"/api/v1/analyze", "/api/v1/research/contributions"}:
            body_limit = MAX_UPLOAD_BYTES + BODY_OVERHEAD_BYTES
        elif path == "/api/v1/analyze/batch":
            body_limit = MAX_BATCH_BYTES + BODY_OVERHEAD_BYTES
        else:
            body_limit = 2 * 1024 * 1024

        request_id = str(uuid.uuid4())
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        if ENVIRONMENT == "production":
            peer_host = (scope.get("client") or ("", 0))[0]
            try:
                peer_ip = ipaddress.ip_address(peer_host)
                trusted_peer = any(peer_ip in network for network in TRUSTED_PROXY_NETWORKS)
            except ValueError:
                trusted_peer = False
            if not trusted_peer:
                response = JSONResponse(
                    status_code=403,
                    content={"detail": {"code": "untrusted_proxy", "message": "Requests must arrive through the configured trusted gateway."}},
                )
                _apply_rejection_headers(response, request_id, path)
                await response(scope, receive, send)
                return
            received_secret = headers.get(b"x-vgrs-proxy-secret", b"")
            expected_secret = PROXY_SHARED_SECRET.encode("utf-8")
            if len(expected_secret) < 32 or not hmac.compare_digest(received_secret, expected_secret):
                response = JSONResponse(
                    status_code=401,
                    content={"detail": {"code": "proxy_auth_required", "message": "The trusted gateway did not authenticate this request."}},
                )
                _apply_rejection_headers(response, request_id, path)
                await response(scope, receive, send)
                return
            role = headers.get(b"x-vgrs-authenticated-role", b"").decode("utf-8", errors="ignore").strip().lower()
            user_id = headers.get(b"x-vgrs-authenticated-user", b"").decode("utf-8", errors="ignore").strip()
            # Liveness is available to the trusted, secret-bearing gateway for probes.
            is_liveness_probe = path == "/api/v1/health" and scope.get("method") == "GET"
            if not is_liveness_probe and (not user_id or role not in {"admin", "analyst", "viewer"}):
                response = JSONResponse(
                    status_code=401,
                    content={"detail": {"code": "identity_required", "message": "The gateway must provide an authenticated user and approved role."}},
                )
                _apply_rejection_headers(response, request_id, path)
                await response(scope, receive, send)
                return
            if scope.get("method") == "POST" and path in {"/api/v1/analyze", "/api/v1/analyze/batch"} and role not in {"admin", "analyst"}:
                response = JSONResponse(
                    status_code=403,
                    content={"detail": {"code": "role_forbidden", "message": "Your authenticated role cannot submit audio for analysis."}},
                )
                _apply_rejection_headers(response, request_id, path)
                await response(scope, receive, send)
                return

        raw_length = headers.get(b"content-length")
        if raw_length:
            try:
                if int(raw_length) > body_limit:
                    response = JSONResponse(
                        status_code=413,
                        content={"detail": {"code": "request_too_large", "message": "Request exceeds the configured request-body limit."}},
                    )
                    _apply_rejection_headers(response, request_id, path)
                    await response(scope, receive, send)
                    return
            except ValueError:
                response = JSONResponse(status_code=400, content={"detail": {"code": "invalid_content_length", "message": "Invalid Content-Length header."}})
                _apply_rejection_headers(response, request_id, path)
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > body_limit:
                    raise RequestBodyTooLarge()
            return message

        response_started = False

        async def secured_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                existing = list(message.get("headers", []))
                lower_names = {name.lower() for name, _ in existing}
                security_headers = {
                    b"x-request-id": request_id.encode("ascii"),
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"DENY",
                    b"referrer-policy": b"no-referrer",
                    b"permissions-policy": b"microphone=(self), camera=(), geolocation=()",
                    b"cross-origin-resource-policy": b"same-origin",
                    b"cross-origin-opener-policy": b"same-origin",
                }
                for name, value in security_headers.items():
                    if name not in lower_names:
                        existing.append((name, value))
                if path.startswith("/api/v1/") and b"cache-control" not in lower_names:
                    existing.append((b"cache-control", b"no-store"))
                if ENVIRONMENT == "production" and b"strict-transport-security" not in lower_names:
                    existing.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                message["headers"] = existing
            await send(message)

        started = time.monotonic()
        try:
            await self.app(scope, limited_receive, secured_send)
        except RequestBodyTooLarge:
            if not response_started:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": {"code": "request_too_large", "message": "Request exceeds the configured request-body limit."}},
                )
                _apply_rejection_headers(response, request_id, path)
                await response(scope, receive, send)
        except Exception:
            if not response_started:
                response = JSONResponse(
                    status_code=500,
                    content={"detail": {"code": "internal_error", "message": "The request failed unexpectedly. Provide the request ID to the operator."}},
                )
                _apply_rejection_headers(response, request_id, path)
                await response(scope, receive, send)
            logger.exception("Unhandled API exception request_id=%s path=%s", request_id, path)
        finally:
            # Deliberately omit filenames, request bodies, query strings and secrets.
            logger.info(
                "request_complete request_id=%s method=%s path=%s elapsed_ms=%d",
                request_id, scope.get("method", "GET"), path, int((time.monotonic() - started) * 1000),
            )


def validate_production_configuration() -> None:
    if ENVIRONMENT != "production":
        return
    if not ORIGINS or "*" in ORIGINS or any(not origin.startswith("https://") for origin in ORIGINS):
        raise RuntimeError("Production startup blocked: VGRS_ALLOWED_ORIGINS must contain explicit HTTPS origins only.")
    if not TRUSTED_HOSTS or "*" in TRUSTED_HOSTS:
        raise RuntimeError("Production startup blocked: set explicit VGRS_TRUSTED_HOSTS.")
    if not TRUSTED_PROXY_NETWORKS or any(
        network.prefixlen < (24 if network.version == 4 else 64)
        for network in TRUSTED_PROXY_NETWORKS
    ):
        raise RuntimeError("Production startup blocked: VGRS_TRUSTED_PROXY_IPS must use IPv4 /24 or narrower, or IPv6 /64 or narrower, for gateway peers.")
    if len(PROXY_SHARED_SECRET) < 32:
        raise RuntimeError("Production startup blocked: set VGRS_PROXY_SHARED_SECRET to a random secret of at least 32 characters.")
    if os.getenv("VGRS_AUTH_MODE", "").strip().lower() != "reverse-proxy":
        raise RuntimeError("Production startup blocked: configure reviewed reverse-proxy authentication (VGRS_AUTH_MODE=reverse-proxy).")
    if os.getenv("VGRS_PROXY_AUTH_ACK", "").strip().lower() != "true":
        raise RuntimeError("Production startup blocked: confirm the reverse proxy authenticates all application routes.")
    # This backend intentionally does not implement user accounts or browser sessions. Do not
    # let a configuration flag masquerade as an authentication implementation.
    if os.getenv("VGRS_PROXY_NETWORK_ISOLATED", "").strip().lower() != "true":
        raise RuntimeError("Production startup blocked: restrict API network access to the authenticated proxy and set VGRS_PROXY_NETWORK_ISOLATED=true.")
    if os.getenv("VGRS_PROXY_RATE_LIMIT_ACK", "").strip().lower() != "true":
        raise RuntimeError("Production startup blocked: configure and confirm rate limiting at the authenticated reverse proxy.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_production_configuration()
    try:
        classifier_runtime.configure()
    except ClassifierConfigurationError as exc:
        logger.error("Classifier configuration error: %s", str(exc))
        raise RuntimeError(str(exc)) from exc
    yield


app = FastAPI(
    title="VGRS — Voice Analysis API",
    version=APP_VERSION,
    description="Acoustic analysis and gated model inference. Voice outputs are estimates, not verified identity or age.",
    lifespan=lifespan,
)
app.add_middleware(RequestGuardMiddleware)
if TRUSTED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Accept", "X-Request-ID"],
)


def _not_configured_prediction(note: str) -> dict[str, Any]:
    return {"status": "model_unavailable", "label": None, "scores": None, "model_version": None, "note": note}


def _analyze(payload: bytes, filename: str) -> AnalysisResponse:
    try:
        result = analyze_audio(
            payload, filename, max_duration=MAX_DURATION, min_duration=MIN_DURATION
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_audio", "message": str(exc)}) from exc

    classification = None
    if result.waveform is not None:
        try:
            classification = classifier_runtime.predict(result.waveform, result.sample_rate)
        except Exception as exc:
            # Acoustic results remain useful; never fabricate classifier output when inference fails.
            logger.warning("Classifier inference failed; prediction withheld (%s).", type(exc).__name__)

    if classification is None:
        unavailable = (
            "No licensed, compatible and validated classification model is enabled. "
            "VGRS does not infer demographic labels from pitch or substitute synthetic predictions."
        )
        vocal_presentation = _not_configured_prediction(unavailable)
        age_group = _not_configured_prediction(unavailable)
    else:
        vocal_presentation = classification["vocal_presentation"]
        age_group = classification["age_group"]

    model_metadata = classifier_runtime.metadata()
    model_info = {
        "feature_pipeline": "vgrs-acoustic-baseline",
        "feature_pipeline_version": APP_VERSION,
        "classification_model": model_metadata.get("version"),
        "mode": MODE,
        "model_enabled": classification is not None,
        "accuracy_established": classifier_runtime.production_ready,
        "classification_status": model_metadata["status"],
        "production_approved": classifier_runtime.production_ready,
    }
    limitations = [
        "Acoustic measurements do not establish gender identity or verified chronological age.",
        "Classification outputs are estimates of perceived vocal presentation and perceived age group.",
        "No production accuracy claim is made without a matching, representative, speaker-independent validation report.",
        "Accent, language, recording equipment, vocal training, health and noise may affect measurements.",
    ]
    if model_metadata.get("limitations"):
        limitations.extend(str(item) for item in model_metadata["limitations"])
    return AnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        status="completed",
        filename=filename,
        audio_quality=result.quality,
        vocal_presentation=vocal_presentation,
        age_group=age_group,
        acoustic_features=result.features,
        model_info=model_info,
        limitations=list(dict.fromkeys(limitations)),
    )


async def _run_analysis(payload: bytes, filename: str) -> AnalysisResponse:
    async with ANALYSIS_SEMAPHORE:
        return await run_in_threadpool(_analyze, payload, filename)


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "VGRS Voice Analysis API", "version": APP_VERSION, "docs": "/docs"}


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    status = classifier_runtime.metadata()["status"]
    return {
        "status": "ok",
        "version": APP_VERSION,
        "mode": MODE,
        "environment": ENVIRONMENT,
        "classification_available": status in {"available", "experimental"},
        "classification_ready": classifier_runtime.production_ready,
    }


@app.get("/api/v1/ready")
def readiness() -> dict[str, Any]:
    metadata = classifier_runtime.metadata()
    production_checks = {
        "environment_configuration": ENVIRONMENT != "production" or (
            bool(ORIGINS) and "*" not in ORIGINS and bool(TRUSTED_HOSTS)
        ),
        "authentication_boundary": ENVIRONMENT != "production" or (
            os.getenv("VGRS_AUTH_MODE", "").strip().lower() == "reverse-proxy"
            and os.getenv("VGRS_PROXY_AUTH_ACK", "").strip().lower() == "true"
            and os.getenv("VGRS_PROXY_NETWORK_ISOLATED", "").strip().lower() == "true"
            and os.getenv("VGRS_PROXY_RATE_LIMIT_ACK", "").strip().lower() == "true"
            and bool(TRUSTED_PROXY_NETWORKS)
            and all(network.prefixlen >= (24 if network.version == 4 else 64) for network in TRUSTED_PROXY_NETWORKS)
            and len(PROXY_SHARED_SECRET) >= 32
        ),
        "licensed_validated_classifier": classifier_runtime.production_ready,
    }
    return {
        "ready": all(production_checks.values()) and (ENVIRONMENT != "production" or classifier_runtime.production_ready),
        "environment": ENVIRONMENT,
        "checks": production_checks,
        "classifier": metadata,
        "note": "HTTP liveness does not imply classification readiness or production acceptance.",
    }


@app.get("/api/v1/models")
def models() -> dict[str, Any]:
    acoustic_model = {
        "id": "vgrs-acoustic-baseline",
        "version": APP_VERSION,
        "kind": "acoustic_features",
        "status": "available",
        "accuracy_established": False,
        "outputs": [
            "f0_median_hz", "f0_p10_hz", "f0_p90_hz", "spectral_centroid_hz",
            "spectral_rolloff_85_hz", "zero_crossing_rate", "active_audio_fraction",
        ],
        "limitations": ["Feature extraction only; not a demographic classifier."],
    }
    return {"models": [acoustic_model, classifier_runtime.metadata()]}


@app.post("/api/v1/research/contributions", status_code=201)
async def create_research_contribution(
    file: UploadFile = File(...),
    consent_to_research: bool = Form(...),
    actual_age_years: int = Form(..., ge=18, le=110),
    participant_code: str = Form(default=""),
) -> dict[str, Any]:
    """Explicitly retain an adult recording for local research; never active in production."""
    if not _research_contributions_enabled():
        raise HTTPException(
            status_code=404,
            detail={"code": "research_contributions_disabled", "message": "Research contributions are disabled. Enable them explicitly in a local development environment."},
        )
    if not consent_to_research:
        await file.close()
        raise HTTPException(
            status_code=403,
            detail={"code": "research_consent_required", "message": "Separate consent to retain this recording for research is required."},
        )

    code = participant_code.strip()
    if code and not re.fullmatch(r"[A-Za-z0-9_-]{4,40}", code):
        await file.close()
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_participant_code", "message": "Use a 4–40 character pseudonymous participant code containing only letters, digits, underscores or hyphens. Do not use a name or email."},
        )

    filename = (file.filename or "upload").replace("\\", "/").split("/")[-1] or "upload"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(status_code=415, detail={"code": "unsupported_format", "message": "Unsupported audio extension."})

    try:
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail={"code": "upload_too_large", "message": "Research recording exceeds the configured per-file upload limit."})

    # Validate the file and run the currently configured model before retaining a copy.
    analysis = await _run_analysis(payload, filename)
    if not analysis.model_info.get("model_enabled") or not analysis.age_group.get("label"):
        raise HTTPException(
            status_code=503,
            detail={"code": "classifier_unavailable", "message": "A loaded age-group model is required to contribute a labeled evaluation example."},
        )

    contribution_id = str(uuid.uuid4())
    contribution_dir = RESEARCH_DATA_DIR / contribution_id
    audio_name = "recording" + ext
    metadata = {
        "schema_version": 1,
        "contribution_id": contribution_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "speaker_id": code or contribution_id,
        "participant_code_provided": bool(code),
        "label_source": "self_reported",
        "consent_to_research": True,
        "consent_text_version": "v1",
        "actual_age_years": int(actual_age_years),
        "actual_age_group": _age_band(float(actual_age_years)),
        "predicted_age_group": analysis.age_group.get("label"),
        "predicted_age_years": analysis.age_group.get("estimated_age_years"),
        "predicted_vocal_presentation": analysis.vocal_presentation.get("label"),
        "model_version": analysis.age_group.get("model_version"),
        "audio_file": audio_name,
        "audio_quality_status": analysis.audio_quality.get("status"),
        "purpose": "offline_research_evaluation_only",
    }

    try:
        contribution_dir.mkdir(parents=True, exist_ok=False)
        # These files live under an ignored local data folder; never include audio
        # in logs, API responses, source control or automatic model updates.
        (contribution_dir / audio_name).write_bytes(payload)
        metadata_path = contribution_dir / "metadata.json"
        temporary_metadata = contribution_dir / "metadata.json.tmp"
        temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary_metadata.replace(metadata_path)
    except OSError as exc:
        shutil.rmtree(contribution_dir, ignore_errors=True)
        logger.error("Research contribution write failed (type=%s)", type(exc).__name__)
        raise HTTPException(status_code=500, detail={"code": "research_storage_failed", "message": "The research contribution could not be stored."}) from exc

    return {
        "contribution_id": contribution_id,
        "stored": True,
        "model_updated": False,
        "model_version": metadata["model_version"],
        "note": "Stored for offline evaluation only. This recording will not change the model automatically. You can delete this contribution using the returned ID.",
    }


@app.delete("/api/v1/research/contributions/{contribution_id}", status_code=204)
def delete_research_contribution(contribution_id: str) -> None:
    """Delete one locally stored research contribution by its unguessable ID."""
    if not _research_contributions_enabled():
        raise HTTPException(status_code=404, detail={"code": "research_contributions_disabled", "message": "Research contributions are disabled."})
    try:
        normalized_id = str(uuid.UUID(contribution_id))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=404, detail={"code": "contribution_not_found", "message": "Research contribution not found."}) from exc
    contribution_dir = RESEARCH_DATA_DIR / normalized_id
    if not contribution_dir.is_dir():
        raise HTTPException(status_code=404, detail={"code": "contribution_not_found", "message": "Research contribution not found."})
    shutil.rmtree(contribution_dir)
    return None


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...), consent: bool = Form(...)) -> AnalysisResponse:
    if not consent:
        await file.close()
        raise HTTPException(status_code=403, detail={"code": "consent_required", "message": "Consent is required before audio analysis."})
    filename = (file.filename or "upload").replace("\\", "/").split("/")[-1] or "upload"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(status_code=415, detail={"code": "unsupported_format", "message": "Unsupported audio extension."})
    try:
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail={"code": "file_too_large", "message": f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB."})
    return await _run_analysis(payload, filename)


@app.post("/api/v1/analyze/batch")
async def analyze_batch(files: list[UploadFile] = File(...), consent: bool = Form(...)) -> dict[str, Any]:
    if not consent:
        for upload in files:
            await upload.close()
        raise HTTPException(status_code=403, detail={"code": "consent_required", "message": "Consent is required before batch audio analysis."})
    if not files:
        raise HTTPException(status_code=422, detail={"code": "empty_batch", "message": "Select at least one audio file."})
    if len(files) > 20:
        for upload in files:
            await upload.close()
        raise HTTPException(status_code=413, detail={"code": "batch_too_large", "message": "Maximum 20 files per batch."})
    aggregate_size = sum(int(upload.size or 0) for upload in files)
    if aggregate_size > MAX_BATCH_BYTES:
        for upload in files:
            await upload.close()
        raise HTTPException(status_code=413, detail={"code": "batch_too_large", "message": f"Combined batch size exceeds {MAX_BATCH_BYTES // (1024*1024)} MB."})

    rows = []
    for upload in files:
        filename = (upload.filename or "upload").replace("\\", "/").split("/")[-1] or "upload"
        try:
            payload = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(payload) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail={"code": "file_too_large", "message": "File exceeds per-recording upload limit."})
            result = await _run_analysis(payload, filename)
            rows.append({"filename": filename, "status": "completed", "result": result.model_dump()})
        except HTTPException as exc:
            rows.append({"filename": filename, "status": "failed", "error": exc.detail})
        except Exception:
            logger.exception("Batch item failed; details withheld from response.")
            rows.append({"filename": filename, "status": "failed", "error": {"code": "analysis_failed", "message": "Analysis failed; see server logs using the request context."}})
        finally:
            await upload.close()
    return {
        "count": len(rows),
        "completed": sum(x["status"] == "completed" for x in rows),
        "failed": sum(x["status"] == "failed" for x in rows),
        "results": rows,
    }
