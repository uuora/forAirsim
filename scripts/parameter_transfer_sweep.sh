#!/usr/bin/env bash
# Sweep fixed parameter vectors through the AirSim plant-swap harness.
# One simulator instance is launched and reused; it is stopped at the end.
set -u
cd "$(dirname "$0")/.." || exit 1

PYTHONPATH=python
export PYTHONPATH
export PYTHONIOENCODING=utf-8
PY=./.venv/python.exe
mkdir -p reports/transfer

launch="--launch --keep-alive"
for cell in \
  "right_angle:default:0.4 0.3 0.2 0.45 1.0 0.70 0.55 1.40 0.20 0.15" \
  "right_angle:mid_fast:0.4 0.3 0.2 0.45 1.0 0.85 0.40 1.10 0.40 0.07" \
  "right_angle:fast:0.4 0.3 0.2 0.45 1.0 1.00 0.25 0.80 0.60 0.00" \
  "right_angle:mid_slow:0.4 0.3 0.2 0.45 1.0 0.50 0.78 1.95 0.12 0.32" \
  "right_angle:slow:0.4 0.3 0.2 0.45 1.0 0.30 1.00 2.50 0.05 0.50" \
  "near_reversal:default:0.4 0.3 0.2 0.45 1.0 0.70 0.55 1.40 0.20 0.15" \
  "near_reversal:mid_fast:0.4 0.3 0.2 0.45 1.0 0.85 0.40 1.10 0.40 0.07" \
  "near_reversal:fast:0.4 0.3 0.2 0.45 1.0 1.00 0.25 0.80 0.60 0.00" \
  "near_reversal:mid_slow:0.4 0.3 0.2 0.45 1.0 0.50 0.78 1.95 0.12 0.32" \
  "near_reversal:slow:0.4 0.3 0.2 0.45 1.0 0.30 1.00 2.50 0.05 0.50"
do
  scenario=${cell%%:*}
  rest=${cell#*:}
  name=${rest%%:*}
  params=${rest#*:}
  output="reports/transfer/${scenario}_${name}"
  rm -rf "$output"
  # shellcheck disable=SC2086
  $PY python/run_airsim_parameter_transfer.py --scenario "$scenario" --parameters $params \
      --output "$output" --max-seconds 60 $launch
  launch=""
done

powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='UE4Editor.exe'\" | Where-Object { \$_.CommandLine -like '*dual_target_control_ue*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null
echo "sweep finished"
