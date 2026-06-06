$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

Set-Location "Z:\Ogurcast"
. .\scripts\set_project_env.ps1
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

python -m pip install torch==2.8.0+cu126 torchaudio==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126

python -m pip install -r requirements.txt

python -m pip install --force-reinstall torch==2.8.0+cu126 torchaudio==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126

python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
$cudaAvailable = python -c "import torch; print('true' if torch.cuda.is_available() else 'false')"
if ($cudaAvailable.Trim() -ne "true") {
    Write-Warning "WARNING: CUDA is not available. WhisperX can run on CPU, but it will be slow."
}
python -c "import whisperx; print('whisperx OK')"
python -c "import fastapi; print('fastapi OK')"
