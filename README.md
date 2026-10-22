# VGRS — Voice Analysis Studio

A local-first browser application for audio validation, acoustic measurements and gated voice-attribute model inference. The default build performs real signal analysis and leaves age-group and perceived-vocal-presentation estimates unavailable until a suitable model is configured. An opt-in public audEERING adapter is provided for **non-commercial research only**; it is experimental and blocked in production. A production classifier must have separate commercial-use rights, a reviewed validation report, and suitable population coverage. The application never fills unavailable predictions with fake labels or scores.

## Start directly on Windows

Prerequisites: Python 3.11, Node.js 22+ (Node 24 has been tested). The backend requirements include `imageio-ffmpeg`, which supplies a bundled FFmpeg executable on supported platforms. Install standalone FFmpeg and add it to `PATH` only if the bundled binary is unavailable on your platform.

1. Open PowerShell and run `cd C:\Projects\VGRS`.
2. Copy `.env.example` to `.env` if you need to change API settings. The defaults are suitable for local development.
3. Install Python dependencies: `python -m venv backend\.venv`, then `backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt`.
4. Install the frontend: `cd frontend`, then `npm install`. The checked-in lockfile supports repeatable installation; use `npm ci` for a clean locked installation.
5. Start the backend from the repository root: `backend\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000`.
6. In another terminal, start the UI: `cd C:\Projects\VGRS\frontend`, then `npm run dev`.

Open the UI at http://127.0.0.1:5173 and API docs at http://127.0.0.1:8000/docs. Frontend API target can be set in `frontend/.env` as `VITE_API_URL=http://127.0.0.1:8000`. The backend binds to loopback by default; do not expose it publicly without security work.

For a convenience startup script, run `powershell -ExecutionPolicy Bypass -File .\scripts\start-local.ps1` from the repository root.

## API contract

- `GET /api/v1/health` — app version, operating mode and classifier readiness.
- `GET /api/v1/models` — feature pipeline and classifier status/limitations.
- `POST /api/v1/analyze` — one multipart file named `file` plus required `consent=true`.
- `POST /api/v1/analyze/batch` — up to 20 multipart files named `files` plus required `consent=true`.
- `GET /docs` — interactive OpenAPI documentation.

Requests without consent are rejected. Defaults: 25 MB per file, 100 MB aggregate batch, up to 20 files, 2–120 seconds, and source sample rates 8–96 kHz. Analysis concurrency is capped at two by default. Request bodies are size-limited, FFmpeg decoding is duration-limited, and API responses carry request IDs and basic security headers. Unsupported extensions return HTTP 415; silent, malformed and too-short decoded recordings return HTTP 422; excess upload size returns HTTP 413. Batch reports retain file-specific status when individual inputs fail. `GET /api/v1/ready` exposes deployment/model readiness separately from `GET /api/v1/health` liveness.

Supported extensions include WAV, FLAC, OGG, AIFF, MP3, M4A and WebM, subject to codec availability. SoundFile/libsndfile handles supported formats first; FFmpeg is used as a fallback for additional codecs. The backend requirements install the `imageio-ffmpeg` package, which provides a bundled FFmpeg executable on supported platforms. Uploaded media is passed to the decoder through stdin and decoded audio is received through stdout; the original upload is not saved as a project file. If the bundled executable is unavailable, install FFmpeg separately and ensure it is available on `PATH`.

## Implemented acoustic measurements

- Mono conversion and actual resampling to 16 kHz.
- Input sample rate, output sample rate, duration, channel count, RMS energy and peak amplitude.
- Active-audio fraction and clipping fraction.
- Spectral centroid, 85% spectral rolloff and zero-crossing rate.
- Basic autocorrelation-based F0 median, 10th percentile, 90th percentile, standard deviation and voiced-frame count.
- Quality warnings for silence/low level, clipping, limited active audio and unavailable pitch tracking.
- Browser file upload, drag/drop, playback, microphone recording, processing consent, single-file analysis, batch analysis and CSV/JSON export.

SNR, jitter, shimmer and word speaking rate are explicitly reported unavailable in this baseline. Basic pitch tracking can fail on noisy, breathy, creaky, unvoiced or polyphonic material. Acoustic characteristics are not proof of identity, gender identity or chronological age.

## Evaluation and model adoption

Install the optional evaluation tools using `backend\.venv\Scripts\python.exe -m pip install -r ml\evaluation\requirements.txt`.

Speaker/duplicate-group-safe data splitting:
`python ml/evaluation/split_by_speaker.py --manifest data/manifest.csv --output data/manifest.with-split.csv --seed 42`

Metrics from a held-out prediction CSV:
`python ml/evaluation/evaluate_predictions.py --predictions data/held-out-predictions.csv --output reports/metrics.json --speaker-col speaker_id --subgroup-col accent_group --bootstrap 1000`

See `ml/evaluation/README.md` for required columns and probability-score formats. Populate duplicate-group identifiers after near-duplicate checks. Do not split recordings from the same speaker across train, validation and test.

Model research and the adoption decision are in `docs/research-decisions.md`. The audEERING public model is non-commercial-only under CC BY-NC-SA 4.0 and the publisher describes the open model as English-focused; it is therefore available only via explicit opt-in research configuration and is blocked in production. VoxProfile lists an MIT license but describes training on Korean-language corpora; that does not establish suitability for multilingual VGRS. Neither is approved as a production model. See `docs/production-readiness.md` for optional research setup, the custom model-plugin contract, validation evidence requirements and fail-closed production deployment gates.

Before enabling classification, confirm legal rights for model weights, code and data; ensure consent and demographic representation; verify label mapping/preprocessing; perform speaker-independent evaluation and subgroup analysis; measure macro-F1, balanced accuracy, per-class precision/recall, confusion matrices, calibration and abstention coverage; document latency, limitations and versioning. Do not use raw neural network scores as calibrated probabilities without evidence.

## Privacy and security

Audio is not intentionally saved to project storage and no audio history is implemented. The API closes each upload after the request. However, multipart parsers may temporarily spool larger request bodies to the operating system temp directory, so this is **not a strict RAM-only guarantee**. Read `docs/privacy.md` before changing deployment boundaries. The frontend has no external font request; recordings are sent to the configured API origin. The consent checkbox and API form field must both be present.

This project is not production-certified. Production mode intentionally refuses startup unless explicit HTTPS origins/hosts, reviewed reverse-proxy authentication and network isolation, proxy-side rate limiting, a licensed custom classifier, and a matching validation report are configured. These environment checks do not install the proxy for you; test the actual deployment boundary independently. Review `docs/privacy.md` and `docs/production-readiness.md` before deployment. Never commit `.env`, private recordings, model secrets, weights or restricted datasets.

## Tests and verification

From the repository root:
- `backend\.venv\Scripts\python.exe -m pytest backend\tests -q`
- `backend\.venv\Scripts\python.exe -m pytest ml\evaluation\tests -q` (requires the optional evaluation dependencies)
- `cd frontend; npm test`
- `npm run build`
- `npm audit --audit-level=moderate`

Reproduce the latest checks on the commit you deploy using the commands above and the GitHub Actions workflow. Tests cover generated tones, API consent and size/duration boundaries, FFmpeg fallback, classifier abstention and fail-closed production gates. Synthetic fixtures do not establish natural-speech or demographic-classifier performance. No real VGRS age/presentation classifier accuracy is claimed; see `docs/evaluation-report.md` and `docs/production-readiness.md` for acceptance limits.

## Documentation map

- `docs/architecture.md` — architecture and status.
- `docs/research-decisions.md` — model candidates, licenses and adoption rationale.
- `docs/model-card.md` — acoustic baseline, limitations and evaluation status.
- `docs/privacy.md` — retention assumptions and deployment safeguards.
- `docs/evaluation-report.md` — evidence and items not yet verified.
- `docs/production-readiness.md` — deployment gates, research-classifier setup and validation contract.
- `data/README.md` — dataset handling policy.
- `ml/evaluation/README.md` — split and metrics commands.
