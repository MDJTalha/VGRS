# VGRS architecture and implementation status

## Request flow

1. The React/Vite browser UI uploads a supported audio file to FastAPI. Browser microphone recording is an explicit user action and requires browser permission.
2. FastAPI applies extension and upload-size checks. The audio pipeline validates decoded samples, sample rate and duration; converts channels to mono; resamples to 16 kHz; calculates conservative audio-quality indicators and acoustic features.
3. The response contains real measured features and a quality status. The current classifier adapter is not configured, so both age-group and perceived-vocal-presentation results explicitly report model unavailable. No synthetic label or confidence score is returned.
4. Batch processing invokes the same single-file analysis pipeline per file and returns successful results and isolated failures together. It is synchronous and limited to 20 recordings per request; it does not use a job queue.

## Components

- `frontend/src/App.tsx`: upload, drag/drop, microphone capture, playback, single and batch analysis, CSV/JSON export and report presentation.
- `backend/app/main.py`: versioned API, upload boundaries, response schemas, CORS and model-readiness metadata.
- `backend/app/audio_pipeline.py`: decode, validate, resample, quality indicators and baseline acoustic measurements.
- `backend/tests`: API and validation tests that use generated tones rather than copyrighted or personal recordings.
- `ml/evaluation`: speaker/group-aware dataset split and metrics reporting utilities.
- `docs`: research decision, model card, privacy and evaluation declarations.

## Boundaries and failure behavior

Unsupported extensions return HTTP 415; malformed, silent or too-short decoded audio returns HTTP 422; over-limit inputs return HTTP 413. A supported extension does not guarantee the decoder has the relevant codec. WAV/FLAC work through SoundFile in a standard installation; additional formats may require local FFmpeg. If model weights are absent, the API remains useful for acoustic analysis and does not imply classification readiness.

The frontend development server is bound to `127.0.0.1:5173`; the API is bound to `127.0.0.1:8000` for local use. This is not an authentication layer and must not be exposed to a public network as-is. Docker files are a convenience for local builds, not a production security review.

## Current maturity

The local baseline has passed its unit/API tests and frontend tests/build on the Windows development machine. A synthetic pure-tone upload verified API connectivity and a measurable F0 result. This does not validate natural-speech pitch tracking, a demographic classifier, fairness, calibration, robustness or deployment security. No classifier is currently enabled, no real-data accuracy is claimed, and no production certification is asserted.
