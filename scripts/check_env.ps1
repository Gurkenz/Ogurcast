$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

Set-Location "Z:\Ogurcast"
. .\scripts\set_project_env.ps1
.\.venv\Scripts\Activate.ps1

Write-Host "HF_TOKEN present: $(if ($env:HF_TOKEN) { 'yes' } else { 'no' })"
Write-Host "HF_HOME: $env:HF_HOME"
Write-Host "HF_HUB_CACHE: $env:HF_HUB_CACHE"
Write-Host "TORCH_HOME: $env:TORCH_HOME"
Write-Host "TMP: $env:TMP"

if (-not $env:OGURCAST_FFMPEG) {
    throw "Error: OGURCAST_FFMPEG is not set."
}
& $env:OGURCAST_FFMPEG -version
python --version
python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -c "import whisperx; print('whisperx OK')"
python -c "from whisperx.diarize import DiarizationPipeline; print('diarization import OK')"
python -c "import fastapi; print('fastapi OK')"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi
} else {
    Write-Host "nvidia-smi not found."
}
