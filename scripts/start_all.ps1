$root = "C:\Users\kitti\Desktop\KrakenSK"
$python = Join-Path $root ".venv\Scripts\python.exe"
$phiProject = "C:\Users\kitti\Projects\kraken-hybrad9"
$phiPython = Join-Path $phiProject "npu-env-real\Scripts\python.exe"
$ollamaExe = "C:\Users\kitti\AppData\Local\Programs\Ollama\ollama.exe"
$lmsExe = Join-Path $env:USERPROFILE ".lmstudio\bin\lms.exe"
$envFile = Join-Path $root ".env"

# --- Clean up all KrakenSK app processes before starting ---
Write-Host "Stopping existing KrakenSK app processes..."
$myPid = $PID
# Kill by window title
cmd /c "taskkill /F /FI `"WINDOWTITLE eq KrakenSK*`" /T" 2>$null | Out-Null
# Kill trader/python processes for this repo and old Phi-3 launches
Get-WmiObject Win32_Process -Filter "Name LIKE 'python%'" |
    Where-Object {
        $_.CommandLine -like "*KrakenSK*" -or
        $_.CommandLine -like "*apps.trader.main*" -or
        $_.CommandLine -like "*scripts\phi3_server.py*"
    } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
# Kill ALL powershell watchdog windows referencing KrakenSK (except this script's own session)
Get-WmiObject Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $myPid -and $_.CommandLine -like "*KrakenSK*" -and $_.CommandLine -like "*NoExit*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 5
Write-Host "App stack clean slate ready."
# ---------------------------------------------------------

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

function Get-UrlPort {
    param(
        [string]$Url,
        [int]$Default = 1234
    )
    try {
        return ([Uri]$Url).Port
    } catch {
        return $Default
    }
}

function Get-LocalModelAliases {
    param(
        [string]$ModelName
    )
    $aliases = New-Object System.Collections.Generic.List[string]
    foreach ($candidate in @(
        $ModelName,
        ($ModelName -replace "-", ""),
        ($ModelName -replace "(?<=[a-z0-9])(?=[A-Z])", "-").ToLower(),
        ($ModelName -replace "(?<=[a-z])(?=\d)", "-").ToLower()
    )) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and -not $aliases.Contains($candidate)) {
            $aliases.Add($candidate)
        }
    }
    return @($aliases)
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
        $result.Message = "Cloud provider configured."
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
    $modelAliases = Get-LocalModelAliases -ModelName $Model
    if ($modelsResp.data) {
        $modelEntry = $modelsResp.data | Where-Object { $modelAliases -contains $_.id } | Select-Object -First 1
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

$replayEnabled = (Get-EnvValue "START_REPLAY_ON_START" "false").ToLower() -eq "true"
$visualPhilEnabled = (Get-EnvValue "START_VISUAL_PHIL_ON_START" "false").ToLower() -eq "true"
$visualPhilFeedEnabled = (Get-EnvValue "START_VISUAL_PHIL_FEED_ON_START" "true").ToLower() -eq "true"
$operatorUiEnabled = (Get-EnvValue "START_OPERATOR_UI_ON_START" "false").ToLower() -eq "true"
$mcpServerEnabled = (Get-EnvValue "START_MCP_SERVER_ON_START" "true").ToLower() -eq "true"
$mcpPublicEnabled = (Get-EnvValue "MCP_PUBLIC_ENABLED" "false").ToLower() -eq "true"
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
$mcpHost = Get-EnvValue "MCP_HOST" "0.0.0.0"
$mcpPort = Get-EnvValue "MCP_PORT" "8765"
$traderDecisionEngine = Get-EnvValue "TRADER_DECISION_ENGINE" "classic"
$policyProfile = Get-EnvValue "POLICY_PROFILE" "custom"
$aggressionMode = Get-EnvValue "AGGRESSION_MODE" "NORMAL"
$advisoryModelProvider = Get-EnvValue "ADVISORY_MODEL_PROVIDER" "local_nemo"
$nemotronProvider = Get-EnvValue "NEMOTRON_PROVIDER" "nvidia"
$nemotronStrategistProvider = Get-EnvValue "NEMOTRON_STRATEGIST_PROVIDER" ""
$nemotronBaseUrl = Get-EnvValue "NEMOTRON_BASE_URL" "http://127.0.0.1:8081"
$nemotronModel = Get-EnvValue "NEMOTRON_MODEL" "nemotron-9b"
$advisoryLocalBaseUrl = Get-EnvValue "ADVISORY_LOCAL_BASE_URL" $nemotronBaseUrl
$localLlmBackend = (Get-EnvValue "LOCAL_LLM_BACKEND" "ollama").ToLower()
$startModelsOnStart = (Get-EnvValue "START_MODELS_ON_START" "true").ToLower() -eq "true"
$nemotronTopCandidateCount = Get-EnvValue "NEMOTRON_TOP_CANDIDATE_COUNT" "15"
$nemotronAllowBuyLowOutsideTop = Get-EnvValue "NEMOTRON_ALLOW_BUY_LOW_OUTSIDE_TOP" "true"
$nemotronAllowBuyMediumOutsideTop = Get-EnvValue "NEMOTRON_ALLOW_BUY_MEDIUM_OUTSIDE_TOP" "true"
$nemotronAllowWatchLowLaneConflict = Get-EnvValue "NEMOTRON_ALLOW_WATCH_LOW_LANE_CONFLICT" "true"
$advisoryMinEntryScore = Get-EnvValue "ADVISORY_MIN_ENTRY_SCORE" "60"
$advisoryMinVolumeRatio = Get-EnvValue "ADVISORY_MIN_VOLUME_RATIO" "1.1"
$nvidiaApiKey = Get-EnvValue "NVIDIA_API_KEY" ""
$nvidiaApiUrl = Get-EnvValue "NVIDIA_API_URL" "https://integrate.api.nvidia.com/v1"
$nvidiaModel = Get-EnvValue "NVIDIA_MODEL" "nvidia/llama-3.3-nemotron-super-49b-v1.5"
$openaiApiKey = Get-EnvValue "OPENAI_API_KEY" ""
$openaiApiUrl = Get-EnvValue "OPENAI_API_URL" "https://api.openai.com/v1"
$openaiModel = Get-EnvValue "OPENAI_MODEL" "gpt-4.1-mini"
$watchdogEnabled = (Get-EnvValue "WATCHDOG_ENABLED" "true").ToLower() -eq "true"
if ([string]::IsNullOrWhiteSpace($nemotronStrategistProvider)) {
    $nemotronStrategistProvider = $nemotronProvider
}

# --- Clean up stale AI model watchdogs/processes from older runs ---
Write-Host "Stopping stale AI model processes that do not match current config..."

# Kill dedicated AI watchdog windows from older start_models runs.
Get-WmiObject Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object {
        $_.ProcessId -ne $myPid -and
        $_.CommandLine -like "*AI-Models*" -and
        $_.CommandLine -like "*NoExit*"
    } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# Phi-3 should not remain alive unless explicitly selected for advisory.
if ($advisoryModelProvider -ne "phi3") {
    Get-WmiObject Win32_Process -Filter "Name LIKE 'python%'" |
        Where-Object {
            $_.ExecutablePath -ieq $phiPython -or
            $_.CommandLine -like "*phi3_server.py*"
        } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

# Stop Ollama only when the current run does not use local Nemotron anywhere.
$needsLocalOllama = ($advisoryModelProvider -eq "local_nemo") -or
    ($nemotronProvider -eq "local") -or
    ($nemotronStrategistProvider -eq "local")
if (-not $needsLocalOllama -or $localLlmBackend -ne "ollama") {
    Get-WmiObject Win32_Process -Filter "Name='ollama.exe'" |
        Where-Object {
            $_.ExecutablePath -ieq $ollamaExe -or
            $_.CommandLine -like "*ollama*serve*"
        } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

Start-Sleep -Seconds 2

$needsPhi3 = $advisoryModelProvider -eq "phi3"
$needsLocalOllama = ($advisoryModelProvider -eq "local_nemo") -or
    ($nemotronProvider -eq "local") -or
    ($nemotronStrategistProvider -eq "local")

if ($startModelsOnStart -and ($needsPhi3 -or $needsLocalOllama)) {
    Write-Host "Starting required AI model services via start_models.ps1..."
    & powershell -ExecutionPolicy Bypass -File (Join-Path $root "scripts\start_models.ps1")
}

$nemotronProbeBaseUrl = if ($nemotronStrategistProvider -eq "local" -and $localLlmBackend -eq "lmstudio") { $advisoryLocalBaseUrl } else { $nemotronBaseUrl }
$nemotronStatus = Test-NemotronBackend -Provider $nemotronStrategistProvider -BaseUrl $nemotronProbeBaseUrl -Model $nemotronModel
if (-not $nemotronStatus.Ready) {
    Write-Warning "$($nemotronStatus.Message) Start Phi-3/Nemotron separately via .\scripts\start_models.ps1 or your external host."
} else {
    Write-Host "Nemotron backend: $($nemotronStatus.Message)"
}

# Check Phi-3 is running before starting the trader
if ($advisoryModelProvider -eq "phi3") {
    Write-Host "Checking Phi-3 (http://127.0.0.1:8084/health)..."
    $phi3Ready = Wait-HttpReady -Url "http://127.0.0.1:8084/health" -TimeoutSec 15
    if (-not $phi3Ready) {
        Write-Warning "Phi-3 not detected on port 8084. Run .\scripts\start_models.ps1 first."
    } else {
        Write-Host "Phi-3 ready."
    }
} else {
    Write-Host "Advisory backend set to $advisoryModelProvider - skipping Phi-3 readiness check."
}

# 1) Universe manager - first run is synchronous so trader starts with a fresh universe
Write-Host "Running universe_manager (this may take 1-3 minutes)..."
$universeRefreshMin = [int](Get-EnvValue "UNIVERSE_REFRESH_INTERVAL_MIN" "5")
$universeProc = Start-Process powershell -ArgumentList "-Command", "cd `"$root`"; & `"$python`" -m apps.universe_manager.main" -PassThru -Wait
Write-Host "Universe ready."

# 1b) Universe refresh loop - auto-restart
if ($watchdogEnabled) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "
        `$host.UI.RawUI.WindowTitle='KrakenSK - universe_manager';
        cd '$root';
        while (`$true) {
            Write-Host '[watchdog] Starting universe_manager...';
            `$env:UNIVERSE_LOOP_MODE='true';
            `$env:UNIVERSE_REFRESH_INTERVAL_MIN='$universeRefreshMin';
            & '$python' -m apps.universe_manager.main;
            Write-Host '[watchdog] universe_manager exited. Restarting in 15s...';
            Start-Sleep -Seconds 15
        }"
} else {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "
        `$host.UI.RawUI.WindowTitle='KrakenSK - universe_manager';
        cd '$root';
        `$env:UNIVERSE_LOOP_MODE='true';
        `$env:UNIVERSE_REFRESH_INTERVAL_MIN='$universeRefreshMin';
        & '$python' -m apps.universe_manager.main;
    "
}
Write-Host "Universe refresh loop started (every ${universeRefreshMin}m)."

# 2) NVIDIA optimizer / review scheduler - auto-restart
if ($watchdogEnabled) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "
        `$host.UI.RawUI.WindowTitle='KrakenSK - review_scheduler';
        cd '$root';
        while (`$true) {
            Write-Host '[watchdog] Starting review_scheduler...';
            `$env:ADVISORY_MODEL_PROVIDER='$advisoryModelProvider';
            `$env:NEMOTRON_PROVIDER='$nemotronProvider';
            `$env:NEMOTRON_STRATEGIST_PROVIDER='$nemotronStrategistProvider';
            `$env:NEMOTRON_BASE_URL='$nemotronBaseUrl';
            `$env:NEMOTRON_MODEL='$nemotronModel';
            `$env:NVIDIA_API_KEY='$nvidiaApiKey';
            `$env:NVIDIA_API_URL='$nvidiaApiUrl';
            `$env:NVIDIA_MODEL='$nvidiaModel';
            `$env:OPENAI_API_KEY='$openaiApiKey';
            `$env:OPENAI_API_URL='$openaiApiUrl';
            `$env:OPENAI_MODEL='$openaiModel';
            & '$python' -m apps.review_scheduler.main;
            Write-Host '[watchdog] review_scheduler exited. Restarting in 15s...';
            Start-Sleep -Seconds 15
        }"
} else {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "
        `$host.UI.RawUI.WindowTitle='KrakenSK - review_scheduler';
        cd '$root';
        `$env:ADVISORY_MODEL_PROVIDER='$advisoryModelProvider';
        `$env:NEMOTRON_PROVIDER='$nemotronProvider';
        `$env:NEMOTRON_STRATEGIST_PROVIDER='$nemotronStrategistProvider';
        `$env:NEMOTRON_BASE_URL='$nemotronBaseUrl';
        `$env:NEMOTRON_MODEL='$nemotronModel';
        `$env:NVIDIA_API_KEY='$nvidiaApiKey';
        `$env:NVIDIA_API_URL='$nvidiaApiUrl';
        `$env:NVIDIA_MODEL='$nvidiaModel';
        `$env:OPENAI_API_KEY='$openaiApiKey';
        `$env:OPENAI_API_URL='$openaiApiUrl';
        `$env:OPENAI_MODEL='$openaiModel';
        & '$python' -m apps.review_scheduler.main;
    "
}

# 3) Trader - auto-restart
if ($watchdogEnabled) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "
        `$host.UI.RawUI.WindowTitle='KrakenSK - trader';
        cd '$root';
        while (`$true) {
            Write-Host '[watchdog] Starting trader...';
            `$env:ADVISORY_MODEL_PROVIDER='$advisoryModelProvider';
            `$env:NEMOTRON_PROVIDER='$nemotronProvider';
            `$env:NEMOTRON_STRATEGIST_PROVIDER='$nemotronStrategistProvider';
            `$env:NEMOTRON_BASE_URL='$nemotronBaseUrl';
            `$env:NEMOTRON_MODEL='$nemotronModel';
            `$env:NVIDIA_API_KEY='$nvidiaApiKey';
            `$env:NVIDIA_API_URL='$nvidiaApiUrl';
            `$env:NVIDIA_MODEL='$nvidiaModel';
            `$env:OPENAI_API_KEY='$openaiApiKey';
            `$env:OPENAI_API_URL='$openaiApiUrl';
            `$env:OPENAI_MODEL='$openaiModel';
            & '$python' -m apps.trader.main;
            Write-Host '[watchdog] Trader exited. Restarting in 10s...';
            Start-Sleep -Seconds 10
        }"
} else {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "
        `$host.UI.RawUI.WindowTitle='KrakenSK - trader';
        cd '$root';
        `$env:ADVISORY_MODEL_PROVIDER='$advisoryModelProvider';
        `$env:NEMOTRON_PROVIDER='$nemotronProvider';
        `$env:NEMOTRON_STRATEGIST_PROVIDER='$nemotronStrategistProvider';
        `$env:NEMOTRON_BASE_URL='$nemotronBaseUrl';
        `$env:NEMOTRON_MODEL='$nemotronModel';
        `$env:NVIDIA_API_KEY='$nvidiaApiKey';
        `$env:NVIDIA_API_URL='$nvidiaApiUrl';
        `$env:NVIDIA_MODEL='$nvidiaModel';
        `$env:OPENAI_API_KEY='$openaiApiKey';
        `$env:OPENAI_API_URL='$openaiApiUrl';
        `$env:OPENAI_MODEL='$openaiModel';
        & '$python' -m apps.trader.main;
    "
}

# 4) Optional replay harness
if ($replayEnabled) {
    $replayCommand = "& `"$python`" -m apps.replay.main --symbols $replaySymbols --start-cash $replayStartCash --warmup-bars $replayWarmupBars --max-steps $replayMaxSteps --summary-path `"$replaySummaryPath`" --trace-path `"$replayTracePath`""
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='KrakenSK - replay'; cd `"$root`"; $replayCommand"
}

# 5) Optional visual Phil server
if ($visualPhilEnabled -and $advisoryModelProvider -ne "phi3") {
    Write-Host "Skipping visual Phi-3 server because advisory backend is $advisoryModelProvider."
} elseif ($visualPhilEnabled) {
    if (-not (Test-Path $phiPython)) {
        Write-Error "Phi-3 Python executable not found at $phiPython"
        exit 1
    }
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='KrakenSK - visual_phil'; cd `"$root`"; `$env:VISUAL_PHI3_PORT='$visualPhilPort'; `$env:VISUAL_PHI3_DEVICE='$visualPhilDevice'; `$env:VISUAL_PHI3_MODEL_DIR='$visualPhilModelDir'; & `"$phiPython`" -m apps.visual_phi3.main"
    [void](Wait-HttpReady -Url "http://127.0.0.1:${visualPhilPort}/health" -TimeoutSec 90)
}

if ($visualPhilFeedEnabled -and $advisoryModelProvider -ne "phi3") {
    Write-Host "Skipping visual Phi-3 feed because advisory backend is $advisoryModelProvider."
} elseif ($visualPhilFeedEnabled) {
    if ($watchdogEnabled) {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "
            `$host.UI.RawUI.WindowTitle='KrakenSK - visual_feed';
            cd '$root';
            while (`$true) {
                Write-Host '[watchdog] Starting visual_feed...';
                `$env:VISUAL_PHI3_REVIEW_URL='$visualPhilReviewUrl';
                `$env:VISUAL_PHI3_INTERVAL_SEC='$visualPhilIntervalSec';
                `$env:VISUAL_PHI3_WINDOW_TITLE='$visualPhilWindowTitle';
                `$env:VISUAL_PHI3_PROCESS_NAME='$visualPhilProcessName';
                `$env:VISUAL_PHI3_PROCESS_PID='$visualPhilProcessPid';
                `$env:VISUAL_OLLAMA_MODEL='$visualOllamaModel';
                `$env:VISUAL_OLLAMA_URL='$visualOllamaUrl';
                & '$python' -m apps.visual_feed.main;
                Write-Host '[watchdog] visual_feed exited. Restarting in 10s...';
                Start-Sleep -Seconds 10
            }"
    } else {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "
            `$host.UI.RawUI.WindowTitle='KrakenSK - visual_feed';
            cd '$root';
            `$env:VISUAL_PHI3_REVIEW_URL='$visualPhilReviewUrl';
            `$env:VISUAL_PHI3_INTERVAL_SEC='$visualPhilIntervalSec';
            `$env:VISUAL_PHI3_WINDOW_TITLE='$visualPhilWindowTitle';
            `$env:VISUAL_PHI3_PROCESS_NAME='$visualPhilProcessName';
            `$env:VISUAL_PHI3_PROCESS_PID='$visualPhilProcessPid';
            `$env:VISUAL_OLLAMA_MODEL='$visualOllamaModel';
            `$env:VISUAL_OLLAMA_URL='$visualOllamaUrl';
            & '$python' -m apps.visual_feed.main;
        "
    }
}

if ($operatorUiEnabled) {
    if ($watchdogEnabled) {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "
            `$host.UI.RawUI.WindowTitle='KrakenSK - operator_ui';
            cd '$root';
            while (`$true) {
                Write-Host '[watchdog] Starting operator_ui...';
                `$env:OPERATOR_UI_HOST='$operatorUiHost';
                `$env:OPERATOR_UI_PORT='$operatorUiPort';
                `$env:ADVISORY_MODEL_PROVIDER='$advisoryModelProvider';
                `$env:NEMOTRON_PROVIDER='$nemotronProvider';
                `$env:NEMOTRON_STRATEGIST_PROVIDER='$nemotronStrategistProvider';
                `$env:NEMOTRON_BASE_URL='$nemotronBaseUrl';
                `$env:NEMOTRON_MODEL='$nemotronModel';
                `$env:NVIDIA_API_KEY='$nvidiaApiKey';
                `$env:NVIDIA_API_URL='$nvidiaApiUrl';
                `$env:NVIDIA_MODEL='$nvidiaModel';
                `$env:OPENAI_API_KEY='$openaiApiKey';
                `$env:OPENAI_API_URL='$openaiApiUrl';
                `$env:OPENAI_MODEL='$openaiModel';
                & '$python' -m apps.operator_ui.main;
                Write-Host '[watchdog] operator_ui exited. Restarting in 10s...';
                Start-Sleep -Seconds 10
            }"
    } else {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "
            `$host.UI.RawUI.WindowTitle='KrakenSK - operator_ui';
            cd '$root';
            `$env:OPERATOR_UI_HOST='$operatorUiHost';
            `$env:OPERATOR_UI_PORT='$operatorUiPort';
            `$env:ADVISORY_MODEL_PROVIDER='$advisoryModelProvider';
            `$env:NEMOTRON_PROVIDER='$nemotronProvider';
            `$env:NEMOTRON_STRATEGIST_PROVIDER='$nemotronStrategistProvider';
            `$env:NEMOTRON_BASE_URL='$nemotronBaseUrl';
            `$env:NEMOTRON_MODEL='$nemotronModel';
            `$env:NVIDIA_API_KEY='$nvidiaApiKey';
            `$env:NVIDIA_API_URL='$nvidiaApiUrl';
            `$env:NVIDIA_MODEL='$nvidiaModel';
            `$env:OPENAI_API_KEY='$openaiApiKey';
            `$env:OPENAI_API_URL='$openaiApiUrl';
            `$env:OPENAI_MODEL='$openaiModel';
            & '$python' -m apps.operator_ui.main;
        "
    }
}

if ($mcpServerEnabled) {
    if ($watchdogEnabled) {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "
            `$host.UI.RawUI.WindowTitle='KrakenSK - mcp_server';
            cd '$root';
            while (`$true) {
                Write-Host '[watchdog] Starting mcp_server...';
                `$env:MCP_HOST='$mcpHost';
                `$env:MCP_PORT='$mcpPort';
                `$env:NEMOTRON_PROVIDER='$nemotronProvider';
                `$env:NEMOTRON_BASE_URL='$nemotronBaseUrl';
                `$env:NEMOTRON_MODEL='$nemotronModel';
                `$env:NVIDIA_API_KEY='$nvidiaApiKey';
                `$env:NVIDIA_API_URL='$nvidiaApiUrl';
                `$env:NVIDIA_MODEL='$nvidiaModel';
                `$env:OPENAI_API_KEY='$openaiApiKey';
                `$env:OPENAI_API_URL='$openaiApiUrl';
                `$env:OPENAI_MODEL='$openaiModel';
                & '$python' -m apps.mcp_server.main;
                Write-Host '[watchdog] mcp_server exited. Restarting in 10s...';
                Start-Sleep -Seconds 10
            }"
    } else {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "
            `$host.UI.RawUI.WindowTitle='KrakenSK - mcp_server';
            cd '$root';
            `$env:MCP_HOST='$mcpHost';
            `$env:MCP_PORT='$mcpPort';
            `$env:NEMOTRON_PROVIDER='$nemotronProvider';
            `$env:NEMOTRON_BASE_URL='$nemotronBaseUrl';
            `$env:NEMOTRON_MODEL='$nemotronModel';
            `$env:NVIDIA_API_KEY='$nvidiaApiKey';
            `$env:NVIDIA_API_URL='$nvidiaApiUrl';
            `$env:NVIDIA_MODEL='$nvidiaModel';
            `$env:OPENAI_API_KEY='$openaiApiKey';
            `$env:OPENAI_API_URL='$openaiApiUrl';
            `$env:OPENAI_MODEL='$openaiModel';
            & '$python' -m apps.mcp_server.main;
        "
    }
    [void](Wait-HttpReady -Url "http://127.0.0.1:${mcpPort}/" -TimeoutSec 30)
}

if ($mcpServerEnabled -and $mcpPublicEnabled) {
    if ($watchdogEnabled) {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "
            `$host.UI.RawUI.WindowTitle='KrakenSK - mcp_tunnel';
            cd '$root';
            while (`$true) {
                Write-Host '[watchdog] Starting mcp_tunnel...';
                & powershell -ExecutionPolicy Bypass -File '.\scripts\start_mcp_tunnel.ps1';
                Write-Host '[watchdog] mcp_tunnel exited. Restarting in 10s...';
                Start-Sleep -Seconds 10
            }"
    } else {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "
            `$host.UI.RawUI.WindowTitle='KrakenSK - mcp_tunnel';
            cd '$root';
            & powershell -ExecutionPolicy Bypass -File '.\scripts\start_mcp_tunnel.ps1';
        "
    }
}

Write-Host "Started KrakenSK app stack:"
Write-Host " - external_ai_services (bring your own Phi-3/Nemotron)"
Write-Host " - universe_manager"
Write-Host " - review_scheduler"
Write-Host " - trader"
Write-Host " - trader_decision_engine ($traderDecisionEngine)"
Write-Host " - policy_profile ($policyProfile)"
Write-Host " - aggression_mode ($aggressionMode)"
Write-Host " - advisory_provider ($advisoryModelProvider)"
Write-Host " - nemotron_provider ($nemotronStrategistProvider)"
Write-Host " - nemotron_model ($nemotronModel)"
if ($nemotronStrategistProvider -eq "openai") {
    Write-Host " - openai_model ($openaiModel)"
}
if ($nemotronStrategistProvider -eq "local") {
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
if ($mcpServerEnabled) {
    Write-Host " - mcp_server (http://127.0.0.1:${mcpPort})"
}
if ($mcpServerEnabled -and $mcpPublicEnabled) {
    Write-Host " - mcp_tunnel (cloudflared)"
}
