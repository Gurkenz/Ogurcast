$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

$ProjectRoot = "Z:\Ogurcast"
New-Item -ItemType Directory -Force -Path $ProjectRoot | Out-Null
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force -Path `
  "scripts", "app", "app\static", "uploads", "outputs", "logs", "models", "models\asr", "models\align", "models\diarization", "tmp", ".cache", ".cache\huggingface", ".cache\torch", ".cache\pip", ".cache\nltk", "tools" | Out-Null

if (-not (Test-Path ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    if ($env:HF_TOKEN) {
        $envContent = Get-Content -LiteralPath ".env" -Raw -Encoding UTF8
        $envContent = $envContent -replace "(?m)^HF_TOKEN=.*$", "HF_TOKEN=$env:HF_TOKEN"
        Set-Content -LiteralPath ".env" -Value $envContent -Encoding UTF8
    } else {
        Write-Warning "Created .env from .env.example. Add HF_TOKEN manually for diarization."
    }
}

if (-not (Test-Path ".\scripts\set_project_env.ps1")) {
    throw "Error: scripts\set_project_env.ps1 is missing."
}

. .\scripts\set_project_env.ps1

if (-not (Test-Path ".venv")) {
    try {
        py -3.11 -m venv .venv
    } catch {
        python -m venv .venv
    }
}

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python --version
pip --version
