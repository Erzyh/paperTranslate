# Build the paperTranslate desktop app -> dist\paperTranslate\paperTranslate.exe
# Usage (repo root):  powershell -ExecutionPolicy Bypass -File desktop\build.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root "backend\.venv\Scripts\python.exe"

Write-Host "[1/4] 화면(frontend)을 빌드합니다..." -ForegroundColor Yellow
Push-Location (Join-Path $root "frontend")
npm ci
if ($LASTEXITCODE -ne 0) { throw "npm ci 실패" }
npm run build
if ($LASTEXITCODE -ne 0) { throw "frontend 빌드 실패" }
Pop-Location

Write-Host "[2/4] 빌드 도구를 확인합니다..." -ForegroundColor Yellow
& $py -m pip install -q -r (Join-Path $root "backend\requirements.txt") -r (Join-Path $root "desktop\requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip 설치 실패" }

Write-Host "[3/4] 레이아웃 인식 / OCR 모델을 준비합니다..." -ForegroundColor Yellow
& $py (Join-Path $root "backend\scripts\fetch_models.py")
if ($LASTEXITCODE -ne 0) { throw "모델 준비 실패" }

Write-Host "[4/4] 앱을 묶습니다 (PyInstaller)..." -ForegroundColor Yellow
& $py -m PyInstaller --noconfirm --clean `
    --distpath (Join-Path $root "dist") `
    --workpath (Join-Path $root "build") `
    (Join-Path $root "desktop\paperTranslate.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 실패" }

Write-Host "완료: dist\paperTranslate\paperTranslate.exe" -ForegroundColor Green
