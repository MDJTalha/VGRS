import asyncio
import ipaddress
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import app.audio_pipeline as audio_pipeline
import app.main as main_module
from app.main import app
from app.audio_pipeline import analyze_audio, AudioValidationError

client = TestClient(app)


def wav_bytes(seconds: float = 3.0, frequency: float = 180.0, amplitude: float = 0.25, sr: int = 16000) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    stream = BytesIO()
    sf.write(stream, (amplitude * np.sin(2 * np.pi * frequency * t)).astype("float32"), sr, format="WAV")
    return stream.getvalue()


def upload(payload: bytes, name: str = "sample.wav", consent: str = "true"):
    return client.post("/api/v1/analyze", data={"consent": consent}, files={"file": (name, payload, "audio/wav")})


def test_health_is_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["classification_ready"] is False


def test_models_disclose_classifier_unavailable():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert response.json()["models"][1]["status"] == "not_configured"


def test_analysis_requires_explicit_consent():
    response = client.post("/api/v1/analyze", files={"file": ("sample.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 403
    denied = upload(wav_bytes(), consent="false")
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "consent_required"


def test_valid_wav_produces_features_but_no_fabricated_predictions():
    response = upload(wav_bytes())
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["acoustic_features"]["f0_median_hz"] is not None
    assert abs(data["acoustic_features"]["f0_median_hz"] - 180) < 8
    assert data["vocal_presentation"]["status"] == "model_unavailable"
    assert data["vocal_presentation"]["label"] is None
    assert data["age_group"]["label"] is None
    assert data["model_info"]["accuracy_established"] is False


def test_resamples_44100_hz_audio_to_16000_hz():
    result = analyze_audio(wav_bytes(sr=44100), "resample.wav")
    assert result.source_sample_rate == 44100
    assert result.sample_rate == 16000
    assert result.features["sample_rate_hz"] == 16000
    assert abs(result.features["f0_median_hz"] - 180) < 8


def test_ffmpeg_fallback_uses_resolved_binary(monkeypatch):
    seen = {}

    def unsupported_metadata(*args, **kwargs):
        raise RuntimeError("Codec is not supported by the primary decoder.")

    def fake_ffmpeg(command, **kwargs):
        seen["command"] = command
        return SimpleNamespace(stdout=wav_bytes(), stderr=b"")

    monkeypatch.setattr(audio_pipeline.sf, "info", unsupported_metadata)
    monkeypatch.setattr(audio_pipeline.subprocess, "run", fake_ffmpeg)
    result = analyze_audio(b"compressed fixture bytes", "voice.m4a")

    assert Path(seen["command"][0]).is_file()
    assert "pipe:0" in seen["command"]
    assert "-t" in seen["command"]
    assert result.features["duration_seconds"] == 3.0
    assert result.waveform.shape[0] == 48000


def test_missing_ffmpeg_returns_actionable_error(monkeypatch):
    monkeypatch.setattr(audio_pipeline, "_ffmpeg_executable", lambda: "vgrs-test-ffmpeg-not-installed")
    with pytest.raises(AudioValidationError, match="FFmpeg is unavailable"):
        analyze_audio(b"not really audio", "voice.mp3")


def test_too_short_audio_is_rejected():
    response = upload(wav_bytes(0.5))
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_audio"


def test_silent_audio_is_rejected():
    assert upload(wav_bytes(amplitude=0.0)).status_code == 422


def test_corrupt_audio_is_rejected():
    assert upload(b"not really audio").status_code == 422


def test_unsupported_extension_is_rejected():
    assert upload(wav_bytes(), "sample.exe").status_code == 415


def test_batch_keeps_success_when_one_recording_fails():
    response = client.post("/api/v1/analyze/batch", data={"consent": "true"}, files=[
        ("files", ("valid.wav", wav_bytes(), "audio/wav")),
        ("files", ("broken.wav", b"garbage", "audio/wav")),
    ])
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["completed"] == 1
    assert data["failed"] == 1


def test_batch_requires_consent():
    response = client.post("/api/v1/analyze/batch", files=[
        ("files", ("valid.wav", wav_bytes(), "audio/wav")),
    ])
    assert response.status_code == 403


def test_decoded_duration_limit_is_checked_from_metadata():
    with pytest.raises(AudioValidationError, match="exceeds the 2-second duration limit"):
        analyze_audio(wav_bytes(seconds=3.0), "too-long.wav", max_duration=2.0)


def test_readiness_and_security_headers_are_explicit():
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["checks"]["licensed_validated_classifier"] is False
    assert data["ready"] is False
    health = client.get("/api/v1/health")
    assert health.headers.get("x-request-id")
    assert health.headers["x-content-type-options"] == "nosniff"



def _middleware_response_status(scope):
    sent = []

    async def downstream(inner_scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    async def run():
        middleware = main_module.RequestGuardMiddleware(downstream)
        await middleware(scope, receive, send)

    asyncio.run(run())
    return next(message["status"] for message in sent if message["type"] == "http.response.start")


def _production_scope(path, method, role=None, secret=None, peer="10.10.0.2"):
    headers = []
    if role:
        headers.append((b"x-vgrs-authenticated-role", role.encode()))
        headers.append((b"x-vgrs-authenticated-user", b"test-operator"))
    if secret:
        headers.append((b"x-vgrs-proxy-secret", secret.encode()))
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": (peer, 12345),
        "server": ("vgrs-api", 8000),
        "state": {},
    }


def test_production_proxy_rejects_untrusted_peer_and_missing_secret(monkeypatch):
    monkeypatch.setattr(main_module, "ENVIRONMENT", "production")
    monkeypatch.setattr(main_module, "TRUSTED_PROXY_NETWORKS", [ipaddress.ip_network("10.10.0.2/32")])
    monkeypatch.setattr(main_module, "PROXY_SHARED_SECRET", "x" * 40)
    untrusted = _production_scope("/api/v1/models", "GET", role="viewer", secret="x" * 40, peer="10.10.0.3")
    no_secret = _production_scope("/api/v1/models", "GET", role="viewer", peer="10.10.0.2")

    assert _middleware_response_status(untrusted) == 403
    assert _middleware_response_status(no_secret) == 401


def test_production_viewer_cannot_analyze_but_analyst_can(monkeypatch):
    monkeypatch.setattr(main_module, "ENVIRONMENT", "production")
    monkeypatch.setattr(main_module, "TRUSTED_PROXY_NETWORKS", [ipaddress.ip_network("10.10.0.2/32")])
    monkeypatch.setattr(main_module, "PROXY_SHARED_SECRET", "x" * 40)
    viewer = _production_scope("/api/v1/analyze", "POST", role="viewer", secret="x" * 40)
    analyst = _production_scope("/api/v1/analyze", "POST", role="analyst", secret="x" * 40)

    assert _middleware_response_status(viewer) == 403
    assert _middleware_response_status(analyst) == 200



def test_production_startup_rejects_broad_proxy_network(monkeypatch):
    monkeypatch.setattr(main_module, "ENVIRONMENT", "production")
    monkeypatch.setattr(main_module, "ORIGINS", ["https://vgrs.example"])
    monkeypatch.setattr(main_module, "TRUSTED_HOSTS", ["api.vgrs.example"])
    monkeypatch.setattr(main_module, "TRUSTED_PROXY_NETWORKS", [ipaddress.ip_network("10.10.0.0/16")])
    monkeypatch.setattr(main_module, "PROXY_SHARED_SECRET", "x" * 40)

    with pytest.raises(RuntimeError, match="VGRS_TRUSTED_PROXY_IPS"):
        main_module.validate_production_configuration()
