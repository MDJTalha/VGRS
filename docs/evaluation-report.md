# Evaluation report

Verification run: 2026-10-22, connected Windows development machine.

## Passed checks

- Backend suite: **11 passed**. Covers health, model registry, explicit consent-required behavior, generated tonal WAV features, F0 tolerance on a synthetic tone, conversion from 44.1 kHz to 16 kHz, rejection of too-short and silent recordings, corrupted content, unsupported extension, and batch partial failure.
- Evaluation tooling suite: **2 passed**. Confirms recordings by one speaker and declared duplicate group are kept in a single partition, and verifies coverage/abstention, probability metrics, subgroup reporting and speaker-cluster bootstrap output.
- Frontend suite: **3 passed**. Verifies workspace heading, explicit unavailability disclosure, and consent checkbox / disabled analysis action by default.
- TypeScript/Vite production build: **passed**.
- NPM dependency audit at moderate severity: **0 vulnerabilities**.
- Current-source live API smoke test: UI and OpenAPI both returned HTTP 200; analysis without consent returned HTTP 403; a consented generated 3-second 180 Hz tone returned completed, acceptable quality, F0 about 181.82 Hz, and model_unavailable for both classification results.
- Optional speaker-split and metrics packages installed and executed in the local backend virtual environment for verification; they remain separately listed in ml/evaluation/requirements.txt.

## Not established

- No representative natural-speech acceptance dataset or speaker-independent demographic-classifier test set was used.
- No age-group or perceived-vocal-presentation classifier is integrated or validated.
- No classification macro-F1, balanced accuracy, per-class precision/recall, calibrated probability, abstention trade-off or subgroup fairness claim is reported for VGRS.
- Synthetic-tone pitch test does not establish accuracy of F0 tracking for ordinary speech.
- All codecs, channel/rate combinations and clipping/noise scenarios have not been individually acceptance-tested. Compressed audio decoding depends on local FFmpeg/libsndfile codec support.
- No physical browser-microphone acceptance test, cross-browser screen-reader audit or full visual regression test has been performed.
- Docker Compose has not been built and exercised as part of these verification runs.
- No production security certification is claimed.

## Reproduce

From the repository root:

- backend\.venv\Scripts\python.exe -m pytest backend\tests -q
- backend\.venv\Scripts\python.exe -m pytest ml\evaluation\tests -q
- cd frontend; npm test
- npm run build
- npm audit --audit-level=moderate

Evaluation tools only measure predictions from provided labeled data; they do not produce a model, verify data licensing or guarantee that a dataset split is representative. Always report dataset and speaker counts, label definitions, confidence intervals and abstention coverage next to classifier metrics.
