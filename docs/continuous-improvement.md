# VGRS controlled continuous improvement

## What this adds

VGRS now has an optional local research-contribution workflow and a repeatable offline age-estimation evaluation/calibration process. It is **not online or automatic self-training**. The production model does not change when a user submits a recording or reports that a prediction was wrong.

The research classifier remains experimental and licensed for non-commercial use only. This workflow is for local, non-commercial research with appropriately consented adult speakers; it must not be used to make production, hiring, identity, eligibility or other high-impact decisions.

## Data collection rules

- Research collection is disabled by default. It is forcibly disabled whenever `VGRS_ENV=production`.
- Normal audio analysis does not persist a recording. A second, separate consent is required to create a research contribution.
- Only adult speaker recordings (self-reported age 18–110) are accepted by this initial workflow. It does not collect or calibrate on minors.
- The submitted age label is self-reported and is not independently verified. Do not describe it as ground truth without a label-quality review.
- The selected file and the age label are copied to the **configured API server's disk** under `data/research-contributions/<random-id>/`. The stored metadata excludes the original filename and has no name/email field. A repeated speaker can optionally use the same random pseudonymous participant code; never use a name, email, or other direct identifier.
- Raw audio is sensitive. The workflow does not encrypt files at application level. Enable it only on a machine/storage location you control, with appropriate OS account permissions and disk protection. Do not expose this local API to the Internet or point the UI at a server you do not trust.
- Each contribution includes a random ID. Delete an individual contribution through the UI's **Delete my contribution** button or `DELETE /api/v1/research/contributions/{contribution_id}`. Deletion from this local folder does not remove copies that a user or operator separately exported.
- Contributions are not uploaded to GitHub by the application, and the data directory / generated research reports are ignored by Git.
- The recording is used for offline research evaluation only; the endpoint returns `model_updated: false`. It does not update model weights or predictions at submission time.

## Enable local research contributions

Only do this after reviewing and accepting the storage implications above. Stop the API, then in the PowerShell session used to start Uvicorn:

```powershell
$env:VGRS_ENV = "development"
$env:VGRS_ENABLE_RESEARCH_CONTRIBUTIONS = "true"
```

Keep the research classifier configured as documented in `docs/production-readiness.md`, then restart the API. The UI's report panel shows the contribution workflow after a single-file analysis. You must confirm the separate research consent and enter the adult speaker's age before the recording is retained. If the API does not have collection enabled, it returns `research_contributions_disabled`; this is intentional.

If you use a custom storage location, set `VGRS_RESEARCH_DATA_DIR` to a directory you control before starting the API.

## Evaluate consented contributions

Run the evaluation from the repository root using the research-enabled backend virtual environment:

```powershell
.\backend\.venv\Scripts\python.exe .\ml\improvement\evaluate_contributions.py
```

The script loads the current six-layer research model, validates each stored consent/label record, re-analyzes its audio, and writes:
- `reports/research-improvement/predictions.csv` — per-contribution current-model age estimates and errors;
- `reports/research-improvement/predictions.summary.json` — overall error, within-five-year rate, age-band accuracy, and per-actual-age-band metrics.

It reports mean absolute error (MAE), median absolute error, within-five-year rate and age-group accuracy. The script does not modify model weights. It will skip malformed/missing contributions without logging their identifiers or audio paths.

## Fit a candidate age calibration

A simple affine calibration can sometimes reduce systematic age-estimation bias. Fit it using the predictions file:

```powershell
.\backend\.venv\Scripts\python.exe .\ml\improvement\fit_age_calibration.py --predictions .\reports\research-improvement\predictions.csv --output .\reports\research-improvement\age-calibration-candidate.json
```

The workflow:
1. Requires at least 30 distinct pseudonymous speakers.
2. Splits by speaker (not by recording) into calibration and held-out evaluation groups.
3. Fits a linear mapping only on the calibration speakers.
4. Compares mean absolute error and age-group accuracy on held-out speakers.
5. Writes a candidate only as a report; it stays `approved_for_research: false`. The script exits with code 2 when the held-out improvement gate fails.

The current promotion gate requires at least 0.5 years MAE improvement and no more than a two-percentage-point drop in held-out age-group accuracy, plus safe parameter bounds. These are lightweight experimental guardrails, not sufficient statistical evidence for production.

## Manually enable an accepted research calibration

Read the JSON candidate and the summary report. Confirm the speaker split, sample size, age distribution, label quality, MAE, age-band accuracy, and any subgroup gaps. If the evidence is adequate for your research use and the candidate passes its gate, manually change `approved_for_research` to `true` in that local ignored JSON file. Then set:

```powershell
$env:VGRS_RESEARCH_AGE_CALIBRATION = "C:\Projects\VGRS\reports\research-improvement\age-calibration-candidate.json"
```

Restart the backend and check `GET /api/v1/models`. The classifier metadata must say `age_calibration.status = active` and report its held-out MAE improvement. If the file is unapproved, malformed, for another model version, has insufficient speaker counts, or fails the accuracy gate, startup fails instead of silently ignoring the calibration.

This calibrator is trained on adult contributions, so runtime applies it only when the raw age estimate is at least 18. It does not calibrate outputs below 18. It remains research-only and is not a commercial/production approval. Do not manually approve a candidate just to force it to load.

## Limits and future stages

This first stage collects opt-in adult examples and creates measurable held-out reports and an optional calibration candidate. It does not fine-tune the underlying wav2vec2 weights, perform automatic deployment, or learn from ordinary analysis requests. Further stages require a commercial-license decision for any paid product, a qualified review of dataset rights and representativeness, stronger uncertainty handling, retention automation and independent validation on a locked speaker-disjoint set. A larger model is not assumed to be better without the same evaluation protocol.
