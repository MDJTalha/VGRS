# Evaluation report

Verification status: updated implementation and test specifications were committed on 2026-10-22. The newest commit has not yet been verified on the connected Windows runtime or confirmed green in CI in this report revision. Run the listed commands on the exact commit before releasing.

## Existing verified baseline (prior to production-hardening changes)

The prior acoustic-only build passed 11 backend tests, two evaluation-tool tests, three frontend tests, a TypeScript/Vite production build, a zero-vulnerability npm audit and a current-source consent/API smoke test. Those results apply to the older baseline commit only, not automatically to the current hardened source.

A synthetic 3-second 180 Hz tone returned populated acoustic features and an explicit unavailable state for both classifiers. This verifies an execution path; it does not validate performance on natural speech.

## Added coverage in the current source (must re-run)

The current source includes additional checks for:
- bundled FFmpeg executable selection and decode fallback;
- metadata-based duration rejection and FFmpeg output duration caps;
- health/readiness distinction, request IDs and response security headers;
- research adapter output mapping, non-finite output rejection and abstention;
- production refusing an unconfigured or non-commercial research classifier;
- custom-classifier label normalization and matching license/validation-report gates;
- Python backend compilation and Python dependency audit in GitHub Actions.

These are implemented tests and checks, not a claim that the current commit has passed them. Run both backend test files and all frontend/build/security checks listed below.

## Still not established

- Real VGRS classification performance: no representative, independently labeled and consented production validation dataset has been supplied or evaluated.
- Commercial model/weight/data licensing for a production classifier.
- Urdu/multilingual suitability and speaker-independent subgroup performance.
- Production macro-F1, balanced accuracy, per-class precision/recall, calibration error, confusion matrix and abstention-coverage acceptance criteria for the actual chosen classifier.
- Real-file acceptance tests for every supported codec, sample-rate/channel combination and malformed input scenario.
- Full browser/device/microphone QA, accessibility/screen-reader review and visual regression.
- Security acceptance of the actual TLS/reverse-proxy authentication, authorization, network isolation, host/CORS configuration, rate limiting, and direct-access bypass controls.
- Restore/rollback, monitoring/alerting, load/soak tests and an operational incident/privacy review.
- Production certification. None is asserted.

## Reproducible local checks

From `C:\\Projects\\VGRS`:

```powershell
git log -1 --oneline
.\\backend\\.venv\\Scripts\\python.exe -m compileall -q backend\\app backend\\tests
.\\backend\\.venv\\Scripts\\python.exe -m pytest backend\\tests -q
.\\backend\\.venv\\Scripts\\python.exe -m pytest ml\\evaluation\\tests -q
Push-Location frontend
npm ci
npm test
npm run build
npm audit --audit-level=moderate
Pop-Location
```

The current CI audit found a vulnerable test-only dependency in the former runtime file; the fix splits runtime and dev dependencies and upgrades pytest. Run `pip-audit -r backend/requirements-dev.txt -r ml/evaluation/requirements.txt` (install `pip-audit` in the environment first) and repeat the codec matrix using authorized sample recordings. Do not mark the dependency audit passed until CI confirms the updated requirement set.

## Production gate

Production mode intentionally refuses startup unless:
- explicit HTTPS origins and trusted hosts are configured;
- a real, reviewed reverse proxy handles TLS, authentication/authorization and rate limiting;
- network rules prevent direct bypass of that proxy;
- a commercially suitable, licensed custom classifier is configured;
- a matching and independently reviewed validation report is present.

These startup checks are configuration guardrails, not a replacement for deployment-level penetration testing or scientific review. See `docs/production-readiness.md`.
