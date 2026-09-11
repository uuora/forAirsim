$ErrorActionPreference = 'Stop'
$source = Join-Path (Split-Path $PSScriptRoot -Parent) 'configs\settings.json'
$targetDir = Join-Path $env:USERPROFILE 'Documents\AirSim'
$target = Join-Path $targetDir 'settings.json'
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
Copy-Item -LiteralPath $source -Destination $target -Force
Write-Output "Installed AirSim settings at $target"
