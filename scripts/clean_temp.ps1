$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

$ProjectRoot = "Z:\Ogurcast"
$TmpPath = Join-Path $ProjectRoot "tmp"
$ResolvedTmp = (Resolve-Path -LiteralPath $TmpPath).Path

if ($ResolvedTmp -ne "$ProjectRoot\tmp") {
    throw "Error: unsafe temp path: $ResolvedTmp"
}

Get-ChildItem -LiteralPath $TmpPath -Force | Remove-Item -Recurse -Force
Write-Host "Temp files cleaned: $ProjectRoot\tmp"
