$phiProject = "C:\Users\kitti\Projects\kraken-hybrad9"
$phiScript = Join-Path $phiProject "phi3_server.py"
$phiPython = Join-Path $phiProject "npu-env-real\Scripts\python.exe"

if (-not (Test-Path $phiScript)) {
    Write-Error "Phi-3 server script not found at $phiScript"
    exit 1
}

if (-not (Test-Path $phiPython)) {
    Write-Error "Phi-3 NPU Python not found at $phiPython"
    exit 1
}

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd `"$phiProject`"; `$env:PHI3_DEVICE='NPU'; & `"$phiPython`" `"$phiScript`""
)

Write-Host "Started Phi-3 server on requested device: NPU"
Write-Host "Project: $phiProject"
