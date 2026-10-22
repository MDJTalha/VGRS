$ErrorActionPreference = "Stop"

# Downloads the publisher's six-layer ONNX model for explicitly non-commercial research.
# This model is NOT approved for a commercial or production VGRS deployment.
$root = Split-Path $PSScriptRoot -Parent
$modelRoot = Join-Path $root "models\audeering-age-gender-6-layer"
$cacheRoot = Join-Path $env:LOCALAPPDATA "VGRS\model-cache"
$archivePath = Join-Path $cacheRoot "w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip"
$url = "https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip"

Write-Host "VGRS research classifier download" -ForegroundColor Cyan
Write-Warning "The publisher licenses these weights for non-commercial use only. Do not use for a commercial deployment."
Write-Host "Model destination: $modelRoot"

if (Test-Path $modelRoot) {
    $existing = @(Get-ChildItem -LiteralPath $modelRoot -Force -ErrorAction SilentlyContinue)
    if ($existing.Count -gt 0 -and -not (Get-ChildItem -LiteralPath $modelRoot -Filter "*.onnx" -Recurse -File -ErrorAction SilentlyContinue)) {
        throw "The model destination already contains files but no ONNX model. Move it aside or choose a clean destination before retrying."
    }
}

New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null

if (!(Test-Path $archivePath) -or (Get-Item $archivePath).Length -lt 1MB) {
    Write-Host "Downloading the official model archive from Zenodo. This is a large download." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $url -OutFile $archivePath
}

try {
    $archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        if ($archive.Entries.Count -eq 0) { throw "The downloaded archive is empty." }
    }
    finally { $archive.Dispose() }
}
catch {
    Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
    throw "The model archive could not be validated as a ZIP file. Retry the download from the publisher source. $($_.Exception.Message)"
}

Expand-Archive -LiteralPath $archivePath -DestinationPath $modelRoot -Force
$onnx = @(Get-ChildItem -LiteralPath $modelRoot -Filter "*.onnx" -Recurse -File)
if ($onnx.Count -eq 0) {
    throw "Extraction completed but no ONNX model was found. Do not enable the adapter; inspect the archive contents and publisher instructions."
}

Write-Host ""
Write-Host "Research model downloaded and extracted." -ForegroundColor Green
Write-Host "ONNX file(s):"
$onnx | ForEach-Object { Write-Host " - $($_.FullName)" }
Write-Host ""
Write-Host "In the SAME PowerShell session used to start the backend, set:" -ForegroundColor Cyan
Write-Host '$env:VGRS_ENV = "development"'
Write-Host '$env:VGRS_CLASSIFIER_BACKEND = "audeering-research"'
Write-Host '$env:VGRS_ENABLE_NONCOMMERCIAL_CLASSIFIER = "true"'
Write-Host ('$env:VGRS_CLASSIFIER_MODEL_DIR = "' + $modelRoot + '"')
Write-Host ""
Write-Host "Then restart Uvicorn and check http://127.0.0.1:8000/api/v1/models."
Write-Host "The model must report experimental. VGRS has not established its accuracy for your target speech population."
