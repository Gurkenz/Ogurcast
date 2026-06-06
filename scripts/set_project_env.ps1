$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

$ProjectRoot = "Z:\Ogurcast"

$env:OGURCAST_ROOT = $ProjectRoot
$env:HF_HOME = "$ProjectRoot\.cache\huggingface"
$env:HF_HUB_CACHE = "$ProjectRoot\.cache\huggingface\hub"
$env:HF_TOKEN_PATH = "$ProjectRoot\.cache\huggingface\token"
$env:TRANSFORMERS_CACHE = "$ProjectRoot\.cache\huggingface\transformers"
$env:TORCH_HOME = "$ProjectRoot\.cache\torch"
$env:PIP_CACHE_DIR = "$ProjectRoot\.cache\pip"
$env:NLTK_DATA = "$ProjectRoot\.cache\nltk"
$env:TMP = "$ProjectRoot\tmp"
$env:TEMP = "$ProjectRoot\tmp"

New-Item -ItemType Directory -Force -Path `
  "$ProjectRoot\.cache\huggingface", `
  "$ProjectRoot\.cache\huggingface\hub", `
  "$ProjectRoot\.cache\huggingface\transformers", `
  "$ProjectRoot\.cache\torch", `
  "$ProjectRoot\.cache\pip", `
  "$ProjectRoot\.cache\nltk", `
  "$ProjectRoot\tmp", `
  "$ProjectRoot\uploads", `
  "$ProjectRoot\outputs", `
  "$ProjectRoot\logs", `
  "$ProjectRoot\models", `
  "$ProjectRoot\models\asr", `
  "$ProjectRoot\models\align", `
  "$ProjectRoot\models\diarization", `
  "$ProjectRoot\tools" | Out-Null

$ffmpegCommand = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
if ($ffmpegCommand) {
    $env:OGURCAST_FFMPEG = $ffmpegCommand.Source
}

$toolsPath = "$ProjectRoot\tools"
$pathParts = $env:PATH -split ";"
if ($pathParts -notcontains $toolsPath) {
    $env:PATH = "$toolsPath;$env:PATH"
}

$envFile = "$ProjectRoot\.env"

if (Test-Path $envFile) {
    Get-Content -LiteralPath $envFile -Encoding UTF8 | ForEach-Object {
        if ($_ -match "^\s*#" -or $_ -match "^\s*$") {
            return
        }

        $parts = $_ -split "=", 2
        if ($parts.Length -eq 2) {
            $name = $parts[0].Trim()
            $value = $parts[1].Trim()
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

Write-Host "Ogurcast environment loaded."
Write-Host "Project root: $ProjectRoot"
Write-Host "HF_HOME: $env:HF_HOME"
Write-Host "TORCH_HOME: $env:TORCH_HOME"
Write-Host "TMP: $env:TMP"
Write-Host "HF_TOKEN present: $([bool]$env:HF_TOKEN)"
