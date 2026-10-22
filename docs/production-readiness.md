# Production readiness and model release gates

## Current release state

VGRS 0.2 adds bounded request handling, a bounded FFmpeg decode duration, a classifier adapter contract, request IDs, response security headers, trusted-host enforcement when configured, and a production startup gate. It is still a local-first development build until the external deployment and model-acceptance requirements below are met.

The built-in acoustic baseline is real signal processing; it is not an age/gender classifier. The public audEERING age/gender model is provided only as an opt-in non-commercial research adapter. It is not enabled by default and cannot run under `VGRS_ENV=production`.

## Development setup

Default development configuration uses `VGRS_CLASSIFIER_BACKEND=none`. This is the safe acoustic-only configuration.

For optional non-commercial research only:

1. Review the source model card and license:
   - https://huggingface.co/audeering/wav2vec2-large-robust-6-ft-age-gender
   - https://github.com/audeering/w2v2-age-gender-how-to
2. Install optional dependencies:
   `backend\\.venv\\Scripts\\python.exe -m pip install -r backend\\requirements-classifier-research.txt`
3. From the repository root, run `powershell -ExecutionPolicy Bypass -File .\\scripts\\download-classifier-research.ps1` to download and extract the official six-layer ONNX model into the ignored local `models\\` folder. The script only downloads model files from the publisher's Zenodo URL; it does not enable the model automatically. Do not commit the model or recordings to Git.
4. Set these variables only for a development or non-commercial research session:
   ```powershell
   $env:VGRS_ENV = "development"
   $env:VGRS_CLASSIFIER_BACKEND = "audeering-research"
   $env:VGRS_ENABLE_NONCOMMERCIAL_CLASSIFIER = "true"
   $env:VGRS_CLASSIFIER_MODEL_DIR = "C:\\Projects\\VGRS\\models\\audeering-age-gender-6-layer"
   ```
5. Restart the backend and check `GET /api/v1/models`. The model must report `experimental`, not `available`. Outputs remain research estimates without calibrated confidence; results are not validated for VGRS, Urdu/multilingual use or representative subgroup performance.

The audEERING public weights are licensed CC BY-NC-SA 4.0 for non-commercial use; the publisher states a separate license is needed for commercial use. The publisher also describes the open model as English-only and less robust than its commercial models. Do not use this adapter in a commercial deployment.

## Production classifier contract

Production must use a separately reviewed classifier plugin. The host imports the configured factory using `VGRS_CLASSIFIER_PLUGIN=python.module:create_classifier`. The factory must return an object with:

- a non-empty `model_version` attribute;
- a synchronous `predict(waveform: numpy.ndarray, sample_rate: int)` method;
- a mapping containing `age_group`, `vocal_presentation`, and matching `model_version`.

Allowed normalized labels:

- `age_group`: `Under 13`, `13–17`, `18–29`, `30–44`, `45–59`, or `60+`;
- `vocal_presentation`: `Feminine-coded`, `Masculine-coded`, `Uncertain`, or `None` to abstain.

Raw scores are not exposed as confidence. Return only calibrated confidence if the plugin/API contract is deliberately extended after calibration testing and UI review. These labels describe estimated age appearance and perceived vocal presentation; they do not establish actual chronological age or gender identity.

Production startup requires `VGRS_CLASSIFIER_LICENSE_APPROVED=true` and a JSON validation report configured with `VGRS_CLASSIFIER_VALIDATION_REPORT`. The report must match the loaded model version and include all of these fields:

- `production_approved: true`, `consent_and_dataset_rights_reviewed: true`;
- `model_version`, `dataset_version`, `evaluation_date`, `label_definition`;
- `speaker_count` of at least 500 distinct held-out speakers;
- `macro_f1`, `balanced_accuracy`, `abstention_coverage` numeric values from 0 to 1;
- non-empty `per_class_metrics`, `subgroup_metrics`, and `calibration` objects.

These are minimum schema gates, not an automatic scientific approval. A qualified reviewer must verify the report, test split integrity, demographic/language representation, rights to data/weights/code, measurement definitions, error rates and deployment use. A model's own README numbers are not VGRS validation evidence. A suitable consented real-speech dataset and an independently reviewed result report are still required.

## Production environment guards

When `VGRS_ENV=production`, startup fails closed unless all of the following are configured:

- `VGRS_ALLOWED_ORIGINS`: explicit HTTPS origins only; no wildcard.
- `VGRS_TRUSTED_HOSTS`: explicit hostnames; no wildcard.
- `VGRS_TRUSTED_PROXY_IPS`: narrow IP/CIDR list for the actual reverse proxy peers. Broad wildcard subnets are rejected.
- `VGRS_PROXY_SHARED_SECRET`: random secret of at least 32 characters, shared only between the gateway and API. The gateway must strip any inbound client-supplied `X-VGRS-Proxy-Secret` header and overwrite it.
- `VGRS_AUTH_MODE=reverse-proxy` and `VGRS_PROXY_AUTH_ACK=true`: all user-facing routes are protected by a reviewed authentication/authorization proxy.
- `VGRS_PROXY_NETWORK_ISOLATED=true`: network rules ensure clients cannot bypass the authenticated proxy to reach FastAPI directly.
- `VGRS_PROXY_RATE_LIMIT_ACK=true`: proxy-side request rate limiting is configured.
- A licensed, matching, validated custom classifier plugin as described above.

The application does not manage accounts or sessions itself. After the shared-secret and trusted-peer checks pass, production requests require proxy-injected `X-VGRS-Authenticated-User` and `X-VGRS-Authenticated-Role` headers. Accepted roles are `admin`, `analyst`, and `viewer`; only admin/analyst may submit analysis requests. The reverse proxy must remove any client-supplied versions of all `X-VGRS-*` headers before injecting its own. The shared secret authenticates the gateway-to-API hop; it does not replace the proxy's user authentication, its authorization rules, or network isolation. Setting acknowledgement variables is not a substitute for installing and testing a real proxy. Never expose Uvicorn directly to the public Internet. Keep it bound to a private interface/loopback and apply TLS, authentication, authorization and rate limiting at the reviewed gateway.

## Resource limits and privacy

- Single audio file: `VGRS_MAX_UPLOAD_MB` (default 25 MB).
- Total batch body: `VGRS_MAX_BATCH_MB` (default 100 MB), up to 20 recordings.
- Duration: `VGRS_MIN_DURATION_SECONDS` (default 2) to `VGRS_MAX_DURATION_SECONDS` (default 120).
- Concurrency: `VGRS_MAX_CONCURRENT_ANALYSES` (default 2).
- SoundFile metadata is checked before decoding when supported. FFmpeg fallback is time-bounded, single-threaded and caps its decoded output duration.
- Request IDs and timing are logged without audio, query strings, filenames or secrets. API responses use security headers and `Cache-Control: no-store`.
- Multipart handling can use operating-system temporary storage; this is not a RAM-only guarantee. Confirm temp-directory permissions, disk encryption, crash dump/log policies and retention for the target host.

## Required release acceptance

A production release is blocked until all items have evidence:

1. API and UI build/test suites pass on the tagged commit.
2. Full codec acceptance tests pass with legitimate, consented fixtures for WAV, FLAC, OGG, AIFF, MP3, M4A and WebM; file duration/size/channel limits and malformed inputs are verified.
3. Browser microphone/upload QA passes on the supported browser/OS matrix.
4. A real classifier license and data-rights decision is signed off; a matching validation report passes independent review.
5. Speaker-independent accuracy, calibration, subgroup performance, out-of-domain abstention and latency/memory are measured for the actual launch languages/populations.
6. A protected deployment with TLS, trusted-host restrictions, authentication/authorization, proxy rate limiting, alerts and log scrubbing is tested.
7. Retention/deletion, consent text, data processing and operational incident procedures are reviewed.
8. Backup/restore (if any state is added), rollback, dependency scanning and an operational owner are defined.

Until then, VGRS is suitable for local development/research and acoustic measurement—not certified production demographic classification.
