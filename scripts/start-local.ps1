$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
if (!(Test-Path ".env")) { Copy-Item ".env.example" ".env" }
if (!(Test-Path "backend\.venv\Scripts\python.exe")) { python -m venv backend\.venv }
& "backend\.venv\Scripts\python.exe" -m pip install -r backend\requirements.txt
Push-Location frontend
npm install
Pop-Location
Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location '$root'; .\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location '$root\frontend'; npm run dev"
Write-Host "API: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "UI:  http://127.0.0.1:5173" -ForegroundColor Green
