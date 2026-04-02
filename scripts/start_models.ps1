$root = "C:\Users\kitti\Desktop\KrakenSK"
$phiProject = "C:\Users\kitti\Projects\kraken-hybrad9"
$phiScript = Join-Path $root "scripts\\phi3_server.py"
$phiPython = Join-Path $phiProject "npu-env-real\Scripts\python.exe"
$ollamaExe = "C:\Users\kitti\AppData\Local\Programs\Ollama\ollama.exe"
$envFile = Join-Path $root ".env"

function Get-EnvValue {
    param(
        [string]$Key,
        [string]$Default = ""
    )
    if (-not (Test-Path $envFile)) {
        return $Default
    }
    $line = Get-Content $envFile | Where-Object { $_ -match "^\s*$Key=" } | Select-Object -First 1
    if (-not $line) {
        return $Default
    }
    return ($line -split "=", 2)[1].Trim()
}

function Wait-HttpReady {
    param(
        [string]$Url,
        [int]$TimeoutSec = 60
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) {
                return $true
            }
        } catch {
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Get-HttpJson {
    param(
        [string]$Url,
        [int]$TimeoutSec = 5
    )
    try {
        return Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec $TimeoutSec -ErrorAction Stop
    } catch {
        return $null
    }
}

if (-not (Test-Path $phiScript)) {
    Write-Error "Phi-3 server script not found at $phiScript"
    exit 1
}

if (-not (Test-Path $phiPython)) {
    Write-Error "Phi-3 Python executable not found at $phiPython"
    exit 1
}

$watchdogEnabled = (Get-EnvValue "WATCHDOG_ENABLED" "true").ToLower() -eq "true"
$advisoryModelProvider = (Get-EnvValue "ADVISORY_MODEL_PROVIDER" "local_nemo").ToLower()
$startPhi3OnStart = (Get-EnvValue "START_PHI3_ON_START" "false").ToLower() -eq "true"
$nemotronProvider = Get-EnvValue "NEMOTRON_PROVIDER" "nvidia"
$nemotronStrategistProvider = Get-EnvValue "NEMOTRON_STRATEGIST_PROVIDER" ""
$nvidiaApiKey = Get-EnvValue "NVIDIA_API_KEY" ""
if ([string]::IsNullOrWhiteSpace($nemotronStrategistProvider)) {
    $nemotronStrategistProvider = $nemotronProvider
}
$nemotronModel = Get-EnvValue "NEMOTRON_MODEL" "nemotron-9b"
$ollamaModelsRoot = Get-EnvValue "OLLAMA_MODELS" "C:\Users\kitti"

Write-Host "Starting persistent AI services only..."

$needsPhi3 = ($advisoryModelProvider -eq "phi3") -or $startPhi3OnStart

if ($needsPhi3) {
    $phi3Already = Wait-HttpReady -Url "http://127.0.0.1:8084/health" -TimeoutSec 3
    if ($phi3Already) {
        Write-Host "Phi-3 already running on port 8084 - skipping start."
    } elseif ($watchdogEnabled) {
        Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", "
            `$host.UI.RawUI.WindowTitle='AI-Models - phi3_npu';
            cd '$root';
            while (`$true) {
                Write-Host '[watchdog] Starting phi3_npu...';
                `$env:PHI3_DEVICE='NPU';
                `$env:PHI3_STRICT_DEVICE='true';
                & '$phiPython' '$phiScript';
                Write-Host '[watchdog] phi3_npu exited. Restarting in 10s...';
                Start-Sleep -Seconds 10
            }"
    } else {
        Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", "
            `$host.UI.RawUI.WindowTitle='AI-Models - phi3_npu';
            cd '$root';
            `$env:PHI3_DEVICE='NPU';
            `$env:PHI3_STRICT_DEVICE='true';
            & '$phiPython' '$phiScript';
        "
    }
    Write-Host "Waiting for Phi-3 to load on NPU (can take 2-3 min)..."
    $phi3Up = Wait-HttpReady -Url "http://127.0.0.1:8084/health" -TimeoutSec 180
    if ($phi3Up) {
        $phi3Health = Get-HttpJson -Url "http://127.0.0.1:8084/health" -TimeoutSec 5
        $phi3Device = ""
        if ($phi3Health -ne $null) {
            $phi3Device = (($phi3Health.device | Out-String).Trim()).ToUpper()
        }
        $phi3StartupOk = $false
        try {
            $phi3SmokeBody = @{
                model = "phi3"
                messages = @(
                    @{ role = "system"; content = "Return strict JSON only." },
                    @{ role = "user"; content = '{"ping":true}' }
                )
                max_tokens = 32
            } | ConvertTo-Json -Compress -Depth 5
            $phi3Smoke = Invoke-RestMethod -Uri "http://127.0.0.1:8084/v1/chat/completions" -Method Post -ContentType "application/json" -Body $phi3SmokeBody -TimeoutSec 30
            if ($phi3Smoke -and $phi3Smoke.choices -and $phi3Smoke.choices.Count -gt 0) {
                $phi3StartupOk = $true
            }
        } catch {
            $phi3StartupOk = $false
        }
        if ($phi3Device -eq "NPU" -and $phi3StartupOk) {
            Write-Host "Phi-3 ready on NPU with chat-completion smoke test passed."
        } elseif (-not [string]::IsNullOrWhiteSpace($phi3Device)) {
            Write-Warning "Phi-3 responded on device '$phi3Device' or failed chat smoke test. Check the phi3_npu window."
        } else {
            Write-Warning "Phi-3 responded but did not report a device. Check the phi3_npu window."
        }
    } else {
        Write-Warning "Phi-3 did not respond within 3 minutes - check the phi3_npu window for errors."
    }
} else {
    Write-Host "Advisory backend set to $advisoryModelProvider - skipping Phi-3 startup."
}

if ($advisoryModelProvider -eq "local_nemo" -or $nemotronStrategistProvider -eq "local") {
    if (-not (Test-Path $ollamaExe)) {
        Write-Error "Ollama executable not found at $ollamaExe"
        exit 1
    }
    $ollamaAlready = $null
    try { $ollamaAlready = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction SilentlyContinue } catch {}
    if ($ollamaAlready) {
        Write-Host "Ollama already running on port 11434 - skipping start."
    } elseif ($watchdogEnabled) {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "
            `$host.UI.RawUI.WindowTitle='AI-Models - ollama';
            while (`$true) {
                Write-Host '[watchdog] Starting ollama...';
                `$env:OLLAMA_KEEP_ALIVE='-1';
                `$env:OLLAMA_MODELS='$ollamaModelsRoot';
                & '$ollamaExe' serve;
                Write-Host '[watchdog] ollama exited. Restarting in 10s...';
                Start-Sleep -Seconds 10
            }"
    } else {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "
            `$host.UI.RawUI.WindowTitle='AI-Models - ollama';
            `$env:OLLAMA_KEEP_ALIVE='-1';
            `$env:OLLAMA_MODELS='$ollamaModelsRoot';
            & '$ollamaExe' serve;
        "
    }
    [void](Wait-HttpReady -Url "http://127.0.0.1:11434/api/tags" -TimeoutSec 60)

    Write-Host "Warming Nemotron model (3 inference passes - first pass may take 5+ min)..."
    $warmPrompt = '{"action":"HOLD"}'
    $warmBody = @{
        model = $nemotronModel
        prompt = $warmPrompt
        stream = $false
        keep_alive = "1440h"
        options = @{ num_predict = 30; temperature = 0; num_ctx = 8192 }
    } | ConvertTo-Json -Compress
    for ($warmPass = 1; $warmPass -le 3; $warmPass++) {
        Write-Host "  Nemo warmup pass $warmPass/3..."
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post -ContentType "application/json" -Body $warmBody -TimeoutSec 360 | Out-Null
            Write-Host "  Pass $warmPass done."
        } catch {
            Write-Warning "  Pass $warmPass failed: $_"
        }
    }
    Write-Host "Nemotron model warm."
}

Write-Host "Started persistent AI services:"
if ($needsPhi3) {
    Write-Host " - phi3_npu"
} else {
    Write-Host " - advisory_backend ($advisoryModelProvider)"
}
if ($advisoryModelProvider -eq "local_nemo" -or $nemotronStrategistProvider -eq "local") {
    Write-Host " - ollama (local nemotron host)"
} else {
    Write-Host " - nemotron_provider ($nemotronStrategistProvider)"
}
