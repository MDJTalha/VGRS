# Privacy and security baseline

## Current local behavior

The intended development deployment is local-only: browser UI at 127.0.0.1:5173 and API at 127.0.0.1:8000. The application does not intentionally save submitted recordings to project storage or a database, has no persistent audio-history feature, and does not log audio samples, filenames or request bodies. Each analysis request requires explicit consent and the API closes each upload after processing.

Multipart parsers such as Starlette may spool uploaded bodies to the operating system's temporary directory while parsing the request. Therefore this version guarantees no intentional permanent application retention, not a strict promise that every incoming byte remains exclusively in RAM. Temporary-file behavior depends on the framework and OS.

## Processing boundaries

- Default maximum size is 25 MB per file, with a 100 MB total request cap for batch analysis.
- Batch requests are limited to 20 recordings.
- Default duration is 2–120 seconds; FFmpeg fallback output is capped just above the configured maximum so the post-decode check can reject overlong input.
- Analysis concurrency is capped at two by default; signal processing runs in a worker thread rather than blocking the ASGI event loop.
- API responses include a request ID, security headers and `Cache-Control: no-store` for versioned API responses.
- Logs deliberately omit uploaded file names, audio, request body, query strings and secrets. Use the request ID to correlate an operator investigation.

## Browser and external requests

Microphone capture requires the consent checkbox and browser microphone permission; microphone tracks stop after capture finishes or the component is disposed. Recording data is sent to the configured API origin. The frontend uses system fonts and does not request an external font service. The default API target is localhost. The optional frontend setting `VITE_API_URL` changes the recipient of recordings and must only point to a service the user has approved and understands.

## Production authentication boundary

The application does not implement user accounts, browser sessions, organization membership or role authorization. Production mode therefore requires an independently reviewed reverse proxy that provides authentication/authorization, rate limiting and TLS; trusted host/origin configuration; and network restrictions that prevent direct access to Uvicorn. The configuration acknowledgement variables are fail-closed guardrails, not a proxy implementation or proof of its security. Do not expose Uvicorn directly to the Internet.

The optional open audEERING classification weights are non-commercial and not approved for a commercial service. Research mode is explicitly gated; production requires a separately licensed classifier and an independently reviewed validation report. Model license decisions and validation state must not be inferred from successful model loading.

## Before production use

- Install and penetration-test authentication, authorization, rate limits, TLS and a reviewed reverse proxy; test that direct API access bypass is impossible.
- Set explicit trusted CORS origins and hostnames. CORS is not authentication.
- Review request timeouts, CPU/memory limits, FFmpeg executable provenance and target-host process isolation.
- Confirm temp-directory permissions, disk encryption, crash dumps, backups, access logs and host telemetry do not violate the retention policy.
- Obtain informed consent before recording or analysis. Training reuse requires separate informed opt-in and governance review.
- Implement deletion and retention controls before adding persistent audio or metadata storage.
- Add secret/dependency scanning, incident response, operational alerts and a deployment/rollback owner.

No audio upload should be reused for model training without explicit, informed opt-in and a reviewed governance basis. The current prototype is not security-certified and is not production-certified.
