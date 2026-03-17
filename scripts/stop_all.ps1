$ErrorActionPreference = "Stop"

$targets = @(
    "KrakenSK - trader",
    "KrakenSK - universe_manager",
    "KrakenSK - phi3_npu",
    "KrakenSK - review_scheduler",
    "KrakenSK - replay",
    "KrakenSK - visual_phil",
    "KrakenSK - visual_feed",
    "KrakenSK - operator_ui",
    "KrakenSK - ollama"
)

$stopped = @()

$processes = Get-Process |
    Where-Object { $_.ProcessName -match '^powershell$|^pwsh$' -and $_.MainWindowTitle }

foreach ($title in $targets) {
    $matches = $processes | Where-Object { $_.MainWindowTitle -eq $title }
    foreach ($proc in $matches) {
        try {
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
            $stopped += [pscustomobject]@{
                ProcessId = $proc.Id
                Window = $title
            }
        } catch {
        }
    }
}

$portTargets = @(8084, 8085, 8780, 11434)
foreach ($port in $portTargets) {
    $lines = netstat -ano | Select-String ":$port\s+.*LISTENING"
    foreach ($line in $lines) {
        $parts = ($line.ToString() -split "\s+") | Where-Object { $_ }
        $pidText = $parts[-1]
        $pid = 0
        if ([int]::TryParse($pidText, [ref]$pid) -and $pid -gt 0) {
            try {
                $proc = Get-Process -Id $pid -ErrorAction Stop
                Stop-Process -Id $pid -Force -ErrorAction Stop
                $stopped += [pscustomobject]@{
                    ProcessId = $pid
                    Window = "port:$port ($($proc.ProcessName))"
                }
            } catch {
            }
        }
    }
}

$root = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $root ".venv\Scripts\python.exe"
$phiPython = "C:\Users\kitti\Projects\kraken-hybrad9\npu-env-real\Scripts\python.exe"
$extraTargets = Get-Process python,pythonw -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -and (
        $_.Path -ieq $projectPython -or
        $_.Path -ieq $phiPython
    )
}

foreach ($proc in $extraTargets) {
    try {
        Stop-Process -Id $proc.Id -Force -ErrorAction Stop
        $stopped += [pscustomobject]@{
            ProcessId = $proc.Id
            Window = "python:$($proc.Path)"
        }
    } catch {
    }
}

$ollamaTargets = Get-Process ollama -ErrorAction SilentlyContinue
foreach ($proc in $ollamaTargets) {
    try {
        Stop-Process -Id $proc.Id -Force -ErrorAction Stop
        $stopped += [pscustomobject]@{
            ProcessId = $proc.Id
            Window = "ollama:$($proc.Path)"
        }
    } catch {
    }
}

if ($stopped.Count -eq 0) {
    Write-Host "No KrakenSK project stack windows found."
    exit 0
}

Write-Host "Stopped KrakenSK project stack windows:"
$stopped | Sort-Object Window, ProcessId | Format-Table -AutoSize
