$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

Set-Location "Z:\Ogurcast"
. .\scripts\set_project_env.ps1
.\.venv\Scripts\Activate.ps1

Write-Host "Starting Ogurcast WhisperX UI..."
Write-Host "Open: http://127.0.0.1:7860"

uvicorn app.main:app --host 127.0.0.1 --port 7860 --reload
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
