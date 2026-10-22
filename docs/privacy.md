# Privacy and security baseline

## Current local behavior

The intended deployment is local-only: browser UI at 127.0.0.1:5173 and API at 127.0.0.1:8000. The application does not intentionally save submitted recordings to project storage or a database, has no audio-history feature, and does not log audio samples. Each request requires explicit consent and the API closes each upload after processing. Compressed formats unsupported by SoundFile can be sent to a local FFmpeg process via stdin and decoded output read from stdout.

Important implementation detail: multipart parsers such as Starlette may spool larger uploads to the operating system's temporary directory while handling a request. Therefore this version guarantees no intentional permanent application retention, not a strict promise that every incoming byte remains exclusively in RAM. Temporary-file behavior depends on the framework and OS. If a strict no-disk policy is required, implement and verify a custom streaming receiver before deploying to that environment.

## Browser and external requests

Microphone capture requires the consent checkbox and browser microphone permission; microphone tracks stop after capture finishes or the component is disposed. Recording data is sent to the configured API endpoint. The stylesheet uses system font fallbacks and does not call an external font service. The default API target is localhost. The optional frontend setting `VITE_API_URL` changes the recipient of recordings and must only point to a service the user has approved and understands.

## Before network exposure

- Add authentication, authorization and rate limits; constrain upload concurrency and duration.
- Set CORS to explicit trusted origins; do not treat CORS as authentication.
- Serve behind TLS and a reviewed reverse proxy. Bind publicly only deliberately.
- Set request timeouts and resource limits; sandbox or restrict FFmpeg invocation where appropriate.
- Add dependency scanning, lock-file review, structured error handling and operational monitoring that never records audio or secrets.
- Review operating-system temp storage, crash dumps, backups, access logs and host-level telemetry for the deployment environment.
- Keep secrets in environment variables or secret storage. Never commit `.env` or personal recordings.
- Obtain informed consent before recording or analysis and separate it from any later research/training use.
- Implement deletion and retention controls before adding persistent audio or metadata storage.

No audio upload should be reused for model training without explicit, informed opt-in and a reviewed governance basis. This prototype is not security-certified.
