param([object]$SourceHashes)
$ErrorActionPreference = 'Stop'
if ($null -eq $SourceHashes) { return }
$freezeWorkspace = Split-Path $PSScriptRoot -Parent
$freezeProperties = @($SourceHashes.PSObject.Properties)
if ($freezeProperties.Count -eq 0) { throw 'Empty declared flight source freeze.' }
foreach ($freezeProperty in $freezeProperties) {
    if ($freezeProperty.Name -notmatch '^[a-zA-Z0-9_]+\.py$' -or [string]$freezeProperty.Value -notmatch '^[0-9a-f]{64}$') {
        throw 'Invalid declared flight source entry.'
    }
    $freezeFile = Join-Path (Join-Path $freezeWorkspace 'python') $freezeProperty.Name
    if (!(Test-Path -LiteralPath $freezeFile -PathType Leaf) -or
        (Get-FileHash -LiteralPath $freezeFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$freezeProperty.Value) {
        throw "Frozen flight source changed before engine launch: $($freezeProperty.Name)"
    }
}
