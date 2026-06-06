param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [Parameter(Mandatory=$false)][string]$OutputDir = "Z:\Ogurcast\outputs",
    [Parameter(Mandatory=$false)][string]$Model = "medium",
    [Parameter(Mandatory=$false)][string]$Language = "ru",
    [Parameter(Mandatory=$false)][string]$Device = "cuda",
    [Parameter(Mandatory=$false)][string]$ComputeType = "float16",
    [Parameter(Mandatory=$false)][int]$BatchSize = 8,
    [Parameter(Mandatory=$false)][int]$MinSpeakers = 2,
    [Parameter(Mandatory=$false)][int]$MaxSpeakers = 2
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

Set-Location "Z:\Ogurcast"
. .\scripts\set_project_env.ps1
.\.venv\Scripts\Activate.ps1

python -m app.whisperx_runner `
  --input "$InputFile" `
  --output-dir "$OutputDir" `
  --model "$Model" `
  --language "$Language" `
  --device "$Device" `
  --compute-type "$ComputeType" `
  --batch-size $BatchSize `
  --diarize `
  --min-speakers $MinSpeakers `
  --max-speakers $MaxSpeakers
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
