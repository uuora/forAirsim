# Incremental preparation for an existing working 03 environment. No flight.
param([switch]$SkipAudit)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskPython = Join-Path $taskRoot '.venv/python.exe'
if (!(Test-Path -LiteralPath $taskPython)) { throw 'Reuse the existing 03 environment at .venv/python.exe first.' }
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $taskRoot 'python'
Push-Location $taskRoot
try {
    & $taskPython -c "import numpy, matplotlib, PyQt5, airsim; print('03/UI dependencies: OK')"
    if ($LASTEXITCODE -ne 0) { throw 'The existing 03 Python environment is incomplete.' }
    & $taskPython -c "import torch; print('Torch:',torch.__version__)"
    if ($LASTEXITCODE -ne 0) { throw 'ABOM additionally needs Torch. Install torch in this same environment, then rerun. See docs/使用说明/跨电脑复现_20261101.md.' }
    & $taskPython scripts/prepare_abom_research.py
    if ($LASTEXITCODE -ne 0) { throw 'Pinned ABOM network preparation failed.' }
    & $taskPython scripts/restore_control_evidence.py
    if ($LASTEXITCODE -ne 0) { throw 'Evidence restoration failed; differing files were not overwritten.' }
    if (!$SkipAudit) {
        & $taskPython scripts/verify-control-reproduction.py
        if ($LASTEXITCODE -ne 0) { throw 'Historical evidence recomputation failed.' }
    }
    Write-Host 'READY: open the trajectory experiment panel. No simulator or flight was started.'
} finally { Pop-Location }
