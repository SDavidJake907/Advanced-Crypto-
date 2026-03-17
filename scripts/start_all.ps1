$root = "C:\Users\kitti\Desktop\KrakenSK"
$python = Join-Path $root ".venv\Scripts\python.exe"
$phiProject = "C:\Users\kitti\Projects\kraken-hybrad9"
$phiScript = Join-Path $phiProject "phi3_server.py"
$phiPython = Join-Path $phiProject "npu-env-real\Scripts\python.exe"
$ollamaExe = "C:\Users\kitti\AppData\Local\Programs\Ollama\ollama.exe"
$envFile = Join-Path $root ".env"

# --- Clean up any existing KrakenSK stack before starting ---
Write-Host "Stopping any existing KrakenSK processes..."
cmd /c "taskkill /F /FI `"WINDOWTITLE eq KrakenSK*`" /T" 2>$null | Out-Null
cmd /c "taskkill /F /IM ollama.exe /T" 2>$null | Out-Null
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Write-Host "Clean slate ready."
# -----------------------------------------------------------

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

function Test-NemotronBackend {
    param(
        [string]$Provider,
        [string]$BaseUrl,
        [string]$Model
    )
    $result = [ordered]@{
        Provider = $Provider
        Ready = $false
        ModelsUrl = ""
        ChatMode = ""
        Message = ""
    }

    if ($Provider -ne "local") {
        $result.Ready = $true
        $result.ChatMode = "cloud"
        $result.Message = "NVIDIA cloud provider configured."
        return [pscustomobject]$result
    }

    $modelsUrl = if ($BaseUrl.TrimEnd('/').EndsWith('/v1')) { "$($BaseUrl.TrimEnd('/'))/models" } else { "$($BaseUrl.TrimEnd('/'))/v1/models" }
    $result.ModelsUrl = $modelsUrl

    try {
        $modelsResp = Invoke-RestMethod -Uri $modelsUrl -Method Get -TimeoutSec 10
    } catch {
        $result.Message = "Nemotron local backend unreachable at $modelsUrl"
        return [pscustomobject]$result
    }

    $modelEntry = $null
    if ($modelsResp.data) {
        $modelEntry = $modelsResp.data | Where-Object { $_.id -eq $Model } | Select-Object -First 1
    }
    if (-not $modelEntry) {
        $available = @()
        if ($modelsResp.data) {
            $available = $modelsResp.data | ForEach-Object { $_.id }
        }
        $result.Message = "Configured Nemotron model not found. Available: $($available -join ', ')"
        return [pscustomobject]$result
    }

    try {
        $showResp = Invoke-RestMethod -Uri "$($BaseUrl.TrimEnd('/'))/api/show" -Method Post -ContentType "application/json" -Body (@{ model = $Model } | ConvertTo-Json -Compress) -TimeoutSec 15
        $capabilities = @()
        if ($showResp.capabilities) {
            $capabilities = @($showResp.capabilities)
        }
        if ($capabilities -contains "completion" -and -not ($capabilities -contains "chat")) {
            $result.ChatMode = "completion-fallback"
            $result.Ready = $true
            $result.Message = "Local Nemotron model is completion-only; runtime will use completion fallback."
            return [pscustomobject]$result
        }
        if ($capabilities -contains "chat") {
            $result.ChatMode = "chat"
            $result.Ready = $true
            $result.Message = "Local Nemotron model supports chat completions."
            return [pscustomobject]$result
        }
        $result.ChatMode = "unknown"
        $result.Ready = $true
        $result.Message = "Local Nemotron model detected; capabilities not explicitly reported."
        return [pscustomobject]$result
    } catch {
        $result.Ready = $true
        $result.ChatMode = "unknown"
        $result.Message = "Local Nemotron model detected; capability probe unavailable."
        return [pscustomobject]$result
    }
}

if (-not (Test-Path $python)) {
    Write-Error "Python executable not found at $python"
    exit 1
}

if (-not (Test-Path $phiScript)) {
    Write-Error "Phi-3 server script not found at $phiScript"
    exit 1
}

if (-not (Test-Path $phiPython)) {
    Write-Error "Phi-3 Python executable not found at $phiPython"
    exit 1
}

$replayEnabled = (Get-EnvValue "START_REPLAY_ON_START" "false").ToLower() -eq "true"
$visualPhilEnabled = (Get-EnvValue "START_VISUAL_PHIL_ON_START" "false").ToLower() -eq "true"
$visualPhilFeedEnabled = (Get-EnvValue "START_VISUAL_PHIL_FEED_ON_START" "true").ToLower() -eq "true"
$operatorUiEnabled = (Get-EnvValue "START_OPERATOR_UI_ON_START" "false").ToLower() -eq "true"
$replaySymbols = Get-EnvValue "REPLAY_SYMBOLS" "BTC/USD DOGE/USD AVAX/USD"
$replayStartCash = Get-EnvValue "REPLAY_START_CASH" "1000"
$replayWarmupBars = Get-EnvValue "REPLAY_WARMUP_BARS" "60"
$replayMaxSteps = Get-EnvValue "REPLAY_MAX_STEPS" "300"
$replaySummaryPath = Get-EnvValue "REPLAY_SUMMARY_PATH" "logs/replay_summary.json"
$replayTracePath = Get-EnvValue "REPLAY_TRACE_PATH" "logs/replay_traces.jsonl"
$visualPhilPort = Get-EnvValue "VISUAL_PHI3_PORT" "8085"
$visualPhilDevice = Get-EnvValue "VISUAL_PHI3_DEVICE" "NPU"
$visualPhilModelDir = Get-EnvValue "VISUAL_PHI3_MODEL_DIR" ""
$visualPhilReviewUrl = Get-EnvValue "VISUAL_PHI3_REVIEW_URL" "http://127.0.0.1:8085/review_image"
$visualPhilIntervalSec = Get-EnvValue "VISUAL_PHI3_INTERVAL_SEC" "30"
$visualOllamaModel = Get-EnvValue "VISUAL_OLLAMA_MODEL" ""
$visualOllamaUrl = Get-EnvValue "VISUAL_OLLAMA_URL" "http://127.0.0.1:11434"
$visualPhilWindowTitle = Get-EnvValue "VISUAL_PHI3_WINDOW_TITLE" "Kraken Desktop"
$visualPhilProcessName = Get-EnvValue "VISUAL_PHI3_PROCESS_NAME" "KrakenDesktop.exe"
$visualPhilProcessPid = Get-EnvValue "VISUAL_PHI3_PROCESS_PID" "0"
$operatorUiHost = Get-EnvValue "OPERATOR_UI_HOST" "127.0.0.1"
$operatorUiPort = Get-EnvValue "OPERATOR_UI_PORT" "8780"
$traderDecisionEngine = Get-EnvValue "TRADER_DECISION_ENGINE" "classic"
$policyProfile = Get-EnvValue "POLICY_PROFILE" "custom"
$nemotronProvider = Get-EnvValue "NEMOTRON_PROVIDER" "nvidia"
$nemotronBaseUrl = Get-EnvValue "NEMOTRON_BASE_URL" "http://127.0.0.1:8081"
$nemotronModel = Get-EnvValue "NEMOTRON_MODEL" "nemotron-9b"
$nemotronTopCandidateCount = Get-EnvValue "NEMOTRON_TOP_CANDIDATE_COUNT" "15"
$nemotronAllowBuyLowOutsideTop = Get-EnvValue "NEMOTRON_ALLOW_BUY_LOW_OUTSIDE_TOP" "true"
$nemotronAllowBuyMediumOutsideTop = Get-EnvValue "NEMOTRON_ALLOW_BUY_MEDIUM_OUTSIDE_TOP" "true"
$nemotronAllowWatchLowLaneConflict = Get-EnvValue "NEMOTRON_ALLOW_WATCH_LOW_LANE_CONFLICT" "true"
$advisoryMinEntryScore = Get-EnvValue "ADVISORY_MIN_ENTRY_SCORE" "60"
$advisoryMinVolumeRatio = Get-EnvValue "ADVISORY_MIN_VOLUME_RATIO" "1.1"
$nvidiaApiKey = Get-EnvValue "NVIDIA_API_KEY" ""
$nvidiaApiUrl = Get-EnvValue "NVIDIA_API_URL" "https://integrate.api.nvidia.com/v1"
$nvidiaModel = Get-EnvValue "NVIDIA_MODEL" "nvidia/llama-3.3-nemotron-super-49b-v1.5"

# 0) Phi-3 advisory server on NPU
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", "`$host.UI.RawUI.WindowTitle='KrakenSK - phi3_npu'; cd `"$phiProject`"; `$env:PHI3_DEVICE='NPU'; & `"$phiPython`" `"$phiScript`""
[void](Wait-HttpReady -Url "http://127.0.0.1:8084/health" -TimeoutSec 90)

# 0b) Local Nemotron host via Ollama, if configured for local mode
if ($nemotronProvider -eq "local" -and (Test-Path $ollamaExe)) {
    $ollamaRunning = Get-Process ollama -ErrorAction SilentlyContinue
    foreach ($proc in $ollamaRunning) {
        try {
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
        } catch {
        }
    }
    Start-Sleep -Seconds 2
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='KrakenSK - ollama'; `$env:OLLAMA_KEEP_ALIVE='-1'; & `"$ollamaExe`" serve"
    [void](Wait-HttpReady -Url "http://127.0.0.1:11434/api/tags" -TimeoutSec 60)
}

$nemotronStatus = Test-NemotronBackend -Provider $nemotronProvider -BaseUrl $nemotronBaseUrl -Model $nemotronModel
if (-not $nemotronStatus.Ready) {
    Write-Warning $nemotronStatus.Message
} else {
    Write-Host "Nemotron backend: $($nemotronStatus.Message)"
}

# 1) Universe manager (one-shot refresh)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='KrakenSK - universe_manager'; cd `"$root`"; & `"$python`" -m apps.universe_manager.main"

# 2) NVIDIA optimizer / review scheduler
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='KrakenSK - review_scheduler'; cd `"$root`"; `$env:NEMOTRON_PROVIDER='$nemotronProvider'; `$env:NEMOTRON_BASE_URL='$nemotronBaseUrl'; `$env:NEMOTRON_MODEL='$nemotronModel'; `$env:NVIDIA_API_KEY='$nvidiaApiKey'; `$env:NVIDIA_API_URL='$nvidiaApiUrl'; `$env:NVIDIA_MODEL='$nvidiaModel'; & `"$python`" -m apps.review_scheduler.main"

# 3) Trader (starts live collector in-process)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='KrakenSK - trader'; cd `"$root`"; `$env:NEMOTRON_PROVIDER='$nemotronProvider'; `$env:NEMOTRON_BASE_URL='$nemotronBaseUrl'; `$env:NEMOTRON_MODEL='$nemotronModel'; `$env:NVIDIA_API_KEY='$nvidiaApiKey'; `$env:NVIDIA_API_URL='$nvidiaApiUrl'; `$env:NVIDIA_MODEL='$nvidiaModel'; & `"$python`" -m apps.trader.main"

# 4) Optional replay harness
if ($replayEnabled) {
    $replayCommand = "& `"$python`" -m apps.replay.main --symbols $replaySymbols --start-cash $replayStartCash --warmup-bars $replayWarmupBars --max-steps $replayMaxSteps --summary-path `"$replaySummaryPath`" --trace-path `"$replayTracePath`""
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='KrakenSK - replay'; cd `"$root`"; $replayCommand"
}

# 5) Optional visual Phil server
if ($visualPhilEnabled) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='KrakenSK - visual_phil'; cd `"$root`"; `$env:VISUAL_PHI3_PORT='$visualPhilPort'; `$env:VISUAL_PHI3_DEVICE='$visualPhilDevice'; `$env:VISUAL_PHI3_MODEL_DIR='$visualPhilModelDir'; & `"$phiPython`" -m apps.visual_phi3.main"
    [void](Wait-HttpReady -Url "http://127.0.0.1:${visualPhilPort}/health" -TimeoutSec 90)
}

if ($visualPhilFeedEnabled) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='KrakenSK - visual_feed'; cd `"$root`"; `$env:VISUAL_PHI3_REVIEW_URL='$visualPhilReviewUrl'; `$env:VISUAL_PHI3_INTERVAL_SEC='$visualPhilIntervalSec'; `$env:VISUAL_PHI3_WINDOW_TITLE='$visualPhilWindowTitle'; `$env:VISUAL_PHI3_PROCESS_NAME='$visualPhilProcessName'; `$env:VISUAL_PHI3_PROCESS_PID='$visualPhilProcessPid'; `$env:VISUAL_OLLAMA_MODEL='$visualOllamaModel'; `$env:VISUAL_OLLAMA_URL='$visualOllamaUrl'; & `"$python`" -m apps.visual_feed.main"
}

if ($operatorUiEnabled) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='KrakenSK - operator_ui'; cd `"$root`"; `$env:OPERATOR_UI_HOST='$operatorUiHost'; `$env:OPERATOR_UI_PORT='$operatorUiPort'; `$env:NEMOTRON_PROVIDER='$nemotronProvider'; `$env:NEMOTRON_BASE_URL='$nemotronBaseUrl'; `$env:NEMOTRON_MODEL='$nemotronModel'; `$env:NVIDIA_API_KEY='$nvidiaApiKey'; `$env:NVIDIA_API_URL='$nvidiaApiUrl'; `$env:NVIDIA_MODEL='$nvidiaModel'; & `"$python`" -m apps.operator_ui.main"
}

Write-Host "Started KrakenSK project stack:"
Write-Host " - phi3_npu"
if ($nemotronProvider -eq "local") {
    Write-Host " - ollama (local nemotron host)"
}
Write-Host " - universe_manager"
Write-Host " - review_scheduler"
Write-Host " - trader"
Write-Host " - trader_decision_engine ($traderDecisionEngine)"
Write-Host " - policy_profile ($policyProfile)"
Write-Host " - nemotron_provider ($nemotronProvider)"
Write-Host " - nemotron_model ($nemotronModel)"
if ($nemotronProvider -eq "local") {
    Write-Host " - nemotron_backend_mode ($($nemotronStatus.ChatMode))"
    Write-Host " - nemotron_status ($($nemotronStatus.Message))"
}
Write-Host " - policy_top_candidates ($nemotronTopCandidateCount)"
Write-Host " - policy_buy_low_outside_top ($nemotronAllowBuyLowOutsideTop)"
Write-Host " - policy_buy_medium_outside_top ($nemotronAllowBuyMediumOutsideTop)"
Write-Host " - policy_watch_low_lane_conflict ($nemotronAllowWatchLowLaneConflict)"
Write-Host " - advisory_min_entry_score ($advisoryMinEntryScore)"
Write-Host " - advisory_min_volume_ratio ($advisoryMinVolumeRatio)"
if ($replayEnabled) {
    Write-Host " - replay"
}
if ($visualPhilEnabled) {
    Write-Host " - visual_phil (:${visualPhilPort})"
}
if ($visualPhilFeedEnabled) {
    Write-Host " - visual_feed"
}
if ($operatorUiEnabled) {
    Write-Host " - operator_ui (http://${operatorUiHost}:${operatorUiPort})"
}
Write-Host "MCP server not touched."
