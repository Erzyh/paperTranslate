# Package the built app for a GitHub release (run desktop\build.ps1 first).
# Produces, in release\:
#   paperTranslate-vX.Y.Z-windows-x64.zip          (top folder "paperTranslate/")
#   paperTranslate-vX.Y.Z-windows-x64.zip.sha256   ("<hex>  <file name>")
# The version comes from backend\app\__init__.py; tag the release "vX.Y.Z".
# The app's self-updater (backend\app\updater.py) relies on these names.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$init = Get-Content (Join-Path $root "backend\app\__init__.py") -Raw
$version = [regex]::Match($init, '__version__\s*=\s*"([^"]+)"').Groups[1].Value
if (-not $version) { throw "backend\app\__init__.py 에서 버전을 찾지 못했습니다" }

$app = Join-Path $root "dist\paperTranslate"
if (-not (Test-Path (Join-Path $app "paperTranslate.exe"))) {
    throw "dist\paperTranslate\paperTranslate.exe 가 없습니다. 먼저 desktop\build.ps1 을 실행하세요"
}

# AGPL-3.0: ship the license text with the app.
Copy-Item (Join-Path $root "LICENSE") (Join-Path $app "LICENSE.txt") -Force

$out = Join-Path $root "release"
New-Item -ItemType Directory -Force $out | Out-Null
$name = "paperTranslate-v$version-windows-x64.zip"
$zip = Join-Path $out $name
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $app -DestinationPath $zip -CompressionLevel Optimal

$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
[IO.File]::WriteAllText("$zip.sha256", "$hash  $name`n")

Write-Host "완료: release\$name ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB)" -ForegroundColor Green
Write-Host "태그 v$version 로 릴리즈를 만들고 두 파일을 올리세요." -ForegroundColor Green
