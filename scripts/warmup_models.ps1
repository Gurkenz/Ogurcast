$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

Set-Location "Z:\Ogurcast"
. .\scripts\set_project_env.ps1
.\.venv\Scripts\Activate.ps1

python -m app.whisperx_runner --warmup --model medium --language ru --device cuda --compute-type float16
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
