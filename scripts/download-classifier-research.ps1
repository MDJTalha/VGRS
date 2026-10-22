$ErrorActionPreference = "Stop"

# Downloads and verifies the publisher's six-layer ONNX model for explicitly
# non-commercial research. This model is NOT approved for a commercial deployment.
$root = Split-Path $PSScriptRoot -Parent
$modelRoot = Join-Path $root "models\audeering-age-gender-6-layer"
$cacheRoot = Join-Path $env:LOCALAPPDATA "VGRS\model-cache"
$archivePath = Join-Path $cacheRoot "w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip"
$python = Join-Path $root "backend\.venv\Scripts\python.exe"
$url = "https://zenodo.org/records/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip"

Write-Host "VGRS research classifier download" -ForegroundColor Cyan
Write-Warning "The publisher licenses these weights for non-commercial use only. Do not use for a commercial deployment."
Write-Host "Model destination: $modelRoot"

if (!(Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Research dependencies are not installed in backend\.venv. Run: .\backend\.venv\Scripts\python.exe -m pip install -r .\backend\requirements-classifier-research.txt"
}

if (Test-Path $modelRoot) {
    $existingModel = @(Get-ChildItem -LiteralPath $modelRoot -Filter "model.onnx" -Recurse -File -ErrorAction SilentlyContinue)
    $existingFiles = @(Get-ChildItem -LiteralPath $modelRoot -Force -ErrorAction SilentlyContinue)
    if ($existingFiles.Count -gt 0 -and $existingModel.Count -eq 0) {
        throw "The model destination contains partial or unrelated files but no model.onnx. Inspect it, then move or remove only this generated folder before retrying: $modelRoot"
    }
}

New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null

# Use the publisher's recommended audeer downloader/extractor and Python's
# zipfile module. This avoids Windows PowerShell 5.1's unloaded ZipFile type.
$pythonScript = @'
import hashlib
import sys
import zipfile
from pathlib import Path

import audeer
import audonnx
import numpy as np

url = sys.argv[1]
cache_root = Path(sys.argv[2])
model_root = Path(sys.argv[3])
expected_archive = Path(sys.argv[4])
expected_md5 = "d1479956f0cf50107198276dc1b4f1d0"
minimum_size = 100 * 1024 * 1024

cache_root.mkdir(parents=True, exist_ok=True)
model_root.mkdir(parents=True, exist_ok=True)

# Remove only a clearly incomplete/corrupt cached archive from this model URL.
if expected_archive.exists():
    cached_is_valid = (
        expected_archive.stat().st_size >= minimum_size
        and zipfile.is_zipfile(expected_archive)
    )
    if not cached_is_valid:
        print(f"Removing incomplete cached archive ({expected_archive.stat().st_size:,} bytes).")
        expected_archive.unlink()

print("Downloading the publisher's six-layer model archive. Expected size is hundreds of MB.")
archive_path = Path(audeer.download_url(url, str(cache_root), verbose=True))
if not archive_path.is_file():
    raise RuntimeError(f"Downloader did not create an archive file: {archive_path}")

size = archive_path.stat().st_size
print(f"Downloaded archive: {archive_path} ({size:,} bytes)")
if size < minimum_size:
    raise RuntimeError(
        f"Download is unexpectedly small ({size:,} bytes); expected the ~333 MB publisher archive. "
        "Do not enable the classifier. Check the network/proxy and retry."
    )
if not zipfile.is_zipfile(archive_path):
    archive_path.unlink(missing_ok=True)
    raise RuntimeError("Downloaded content is not a valid ZIP archive. The incomplete file was removed.")

digest = hashlib.md5(archive_path.read_bytes()).hexdigest()
if digest.lower() != expected_md5:
    archive_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"Publisher archive checksum mismatch (MD5 {digest}). The file was removed; retry the download."
    )

with zipfile.ZipFile(archive_path) as archive:
    if not archive.namelist():
        raise RuntimeError("Publisher archive contains no files.")

print("Archive checksum verified. Extracting model...")
audeer.extract_archive(str(archive_path), str(model_root))

onnx_files = list(model_root.rglob("model.onnx"))
yaml_files = list(model_root.rglob("model.yaml"))
if not onnx_files or not yaml_files:
    raise RuntimeError(
        "Extraction did not produce both model.onnx and model.yaml. "
        f"Inspect the extracted folder: {model_root}"
    )

print("Loading the extracted model with audonnx...")
model = audonnx.load(str(model_root))
signal = np.random.default_rng(0).normal(0.0, 0.01, size=32000).astype(np.float32)
outputs = model(signal, 16000)
age = np.asarray(outputs.get("logits_age"))
gender = np.asarray(outputs.get("logits_gender"))
if age.size != 1 or gender.size != 3:
    raise RuntimeError(
        f"Unexpected model output shapes: logits_age={age.shape}, logits_gender={gender.shape}"
    )

print("Model smoke test passed (this verifies loading/output shape, not prediction accuracy).")
print("Model directory:", model_root)
print("ONNX file(s):")
for path in onnx_files:
    print(" -", path)
'@

$pythonScript | & $python - $url $cacheRoot $modelRoot $archivePath
if ($LASTEXITCODE -ne 0) {
    throw "Research model download or verification failed. The classifier remains disabled; inspect the Python error above."
}

Write-Host ""
Write-Host "Research model downloaded, checksum-verified, extracted, and load-tested." -ForegroundColor Green
Write-Host "In the SAME PowerShell session used to start the backend, set:" -ForegroundColor Cyan
Write-Host '$env:VGRS_ENV = "development"'
Write-Host '$env:VGRS_CLASSIFIER_BACKEND = "audeering-research"'
Write-Host '$env:VGRS_ENABLE_NONCOMMERCIAL_CLASSIFIER = "true"'
Write-Host ('$env:VGRS_CLASSIFIER_MODEL_DIR = "' + $modelRoot + '"')
Write-Host ""
Write-Host "Stop any old VGRS API process on port 8000, then restart Uvicorn."
Write-Host "Check http://127.0.0.1:8000/api/v1/models and confirm classification status is experimental."
Write-Host "VGRS has not established this research model's accuracy for your target speech population."
