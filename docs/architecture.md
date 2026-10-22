# VGRS architecture and implementation status

## Request flow

1. The React/Vite browser UI uploads a supported audio file to FastAPI. Browser microphone recording is explicit and requires browser permission and user consent.
2. Request guard middleware assigns a request ID, enforces request-body bounds, and applies security headers. The API checks file count, per-file and total-batch sizes, consent, extension and duration limits.
3. SoundFile metadata is checked before materializing decoded arrays when the format is supported. Compressed formats can use the bundled/system FFmpeg fallback with a bounded output duration, one FFmpeg thread and a timeout.
4. The signal pipeline converts supported input to mono and resamples to 16 kHz. It calculates recording-quality indicators and actual acoustic features.
5. The optional classifier runtime is disabled by default. The non-commercial audEERING research adapter is opt-in and blocked in production. A production custom plugin needs a reviewed license decision and matching, production-approved validation report. An unconfigured or failing classifier never prevents a valid acoustic report from being returned and never generates synthetic labels.
6. Batch processing invokes the same single-file pipeline per file and retains independent failures. It is bounded by file count and aggregate request size. Analysis runs outside the async event loop and concurrency is capped.

## Components

- `frontend/src/App.tsx`: upload, drag/drop, microphone capture, playback, single and batch analysis, CSV/JSON export and report presentation.
- `backend/app/main.py`: versioned API, resource boundaries, production startup gate, readiness/liveness endpoints, security headers, request IDs, CORS and model metadata.
- `backend/app/audio_pipeline.py`: metadata preflight, decode, validation, resampling, quality indicators and acoustic measurements.
- `backend/app/classifier.py`: normalized adapter contract, opt-in non-commercial research model, abstention behavior and production approval gates.
- `backend/tests`: API, signal, decode boundary, readiness and classifier-contract tests using generated tones and fake adapters.
- `ml/evaluation`: speaker/group-aware dataset split and metrics reporting utilities.
- `docs/production-readiness.md`: deployment gates, optional research setup, custom model contract and release acceptance requirements.

## API readiness signals

- `GET /api/v1/health` is a liveness/metadata endpoint. A live API is not proof that a classifier is available.
- `GET /api/v1/models` lists the acoustic baseline and classifier's configured state.
- `GET /api/v1/ready` exposes production-approval checks separately. In development with no classifier, the app can continue serving acoustic features while readiness remains false.

## Production boundary

When `VGRS_ENV=production`, startup fails unless HTTPS origins/hosts, narrow trusted-proxy IP networks, a 32+ character proxy shared secret, gateway-injected authenticated-user/role headers, proxy authentication/authorization, rate limiting and network isolation are configured, and a licensed custom classifier has a matching production-approved report. The API enforces the trusted network + secret, allows only admin/analyst/viewer roles, and requires admin/analyst for analysis POSTs.

The public audEERING weights are only used in an explicitly enabled non-commercial research environment and are never accepted by the production configuration. The repository does not contain any third-party model weights or private voice data.

## Current maturity

The code provides real acoustic measurements, compressed format fallback, bounded request/decode behavior and a fail-closed model interface. A successful synthetic-tune test validates a code path only. No VGRS production age/presentation classification accuracy has been established. A real consented, speaker-independent data set, license review, model-specific subgroup/calibration evidence and protected target deployment are still release prerequisites. No production certification is asserted.
