$root = "C:\Users\kitti\Desktop\KrakenSK"
$phiProject = "C:\Users\kitti\Projects\kraken-hybrad9"
$phiScript = Join-Path $root "scripts\\phi3_server.py"
$phiPython = Join-Path $phiProject "npu-env-real\Scripts\python.exe"
$ollamaExe = "C:\Users\kitti\AppData\Local\Programs\Ollama\ollama.exe"
$lmsExe = Join-Path $env:USERPROFILE ".lmstudio\bin\lms.exe"
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

function Get-WslCommandLiteral {
    param(
        [string]$Value
    )
    return $Value
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
$nemotronBaseUrl = Get-EnvValue "NEMOTRON_BASE_URL" "http://127.0.0.1:11434"
$advisoryLocalBaseUrl = Get-EnvValue "ADVISORY_LOCAL_BASE_URL" $nemotronBaseUrl
$advisoryLocalModel = Get-EnvValue "ADVISORY_LOCAL_MODEL" ""
$localLlmBackend = (Get-EnvValue "LOCAL_LLM_BACKEND" "ollama").ToLower()
$nvidiaApiKey = Get-EnvValue "NVIDIA_API_KEY" ""
if ([string]::IsNullOrWhiteSpace($nemotronStrategistProvider)) {
    $nemotronStrategistProvider = $nemotronProvider
}
$nemotronModel = Get-EnvValue "NEMOTRON_MODEL" "nemotron-9b"
$localLlmLoadKey = Get-EnvValue "LOCAL_LLM_LOAD_KEY" $nemotronModel
$ollamaModelsRoot = Get-EnvValue "OLLAMA_MODELS" "C:\Users\kitti"
$localLlmBaseUrl = if (-not [string]::IsNullOrWhiteSpace($advisoryLocalBaseUrl)) { $advisoryLocalBaseUrl } else { $nemotronBaseUrl }
$localLlmPort = Get-UrlPort -Url $localLlmBaseUrl -Default 1234
$vllmWslDistro = Get-EnvValue "VLLM_WSL_DISTRO" "Ubuntu"
$vllmWslHome = Get-EnvValue "VLLM_WSL_HOME" "/home/$env:USERNAME"
$vllmWslVenv = Get-EnvValue "VLLM_WSL_VENV" "$vllmWslHome/.venvs/vllm"
$vllmWslModel = Get-EnvValue "VLLM_WSL_MODEL" "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
$vllmWslPort = Get-EnvValue "VLLM_WSL_PORT" "8000"
$vllmWslHost = Get-EnvValue "VLLM_WSL_HOST" "0.0.0.0"
$vllmWslExtraArgs = Get-EnvValue "VLLM_WSL_EXTRA_ARGS" ""
$vllmWslReadyUrl = "http://127.0.0.1:$vllmWslPort/v1/models"

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
    if ($localLlmBackend -eq "lmstudio") {
        if (-not (Test-Path $lmsExe)) {
            Write-Error "LM Studio CLI not found at $lmsExe"
            exit 1
        }
        $modelsUrl = if ($localLlmBaseUrl.TrimEnd('/').EndsWith('/v1')) { "$($localLlmBaseUrl.TrimEnd('/'))/models" } else { "$($localLlmBaseUrl.TrimEnd('/'))/v1/models" }
        $lmStudioReady = Wait-HttpReady -Url $modelsUrl -TimeoutSec 3
        if ($lmStudioReady) {
            Write-Host "LM Studio server already running at $localLlmBaseUrl - skipping start."
        } else {
            Write-Host "Starting LM Studio server on port $localLlmPort..."
            & $lmsExe server start --port $localLlmPort | Out-Null
        }
        [void](Wait-HttpReady -Url $modelsUrl -TimeoutSec 60)

        $loadedText = (& $lmsExe ps 2>&1 | Out-String)
        $strategistAliases = @(
            Get-LocalModelAliases -ModelName $nemotronModel
            Get-LocalModelAliases -ModelName $localLlmLoadKey
        ) | Select-Object -Unique
        $strategistLoaded = $false
        foreach ($alias in $strategistAliases) {
            if ($loadedText -match [regex]::Escape($alias)) {
                $strategistLoaded = $true
                break
            }
        }
        if (-not $strategistLoaded) {
            Write-Host "Loading local strategist model in LM Studio..."
            & $lmsExe load $localLlmLoadKey --identifier $nemotronModel --gpu max -y | Out-Null
            $loadedText = (& $lmsExe ps 2>&1 | Out-String)
        } else {
            Write-Host "LM Studio model $nemotronModel already loaded."
        }
        if (
            $advisoryModelProvider -eq "local_nemo" -and
            -not [string]::IsNullOrWhiteSpace($advisoryLocalModel) -and
            $advisoryLocalModel -ne $nemotronModel -and
            $loadedText -notmatch [regex]::Escape($advisoryLocalModel)
        ) {
            Write-Host "Loading advisory local model in LM Studio..."
            & $lmsExe load $advisoryLocalModel --identifier $advisoryLocalModel --gpu max -y | Out-Null
        }
        Write-Host "LM Studio local model host ready."
    } elseif ($localLlmBackend -eq "vllm_wsl") {
        $vllmReady = Wait-HttpReady -Url $vllmWslReadyUrl -TimeoutSec 3
        if ($vllmReady) {
            Write-Host "WSL vLLM server already running at http://127.0.0.1:$vllmWslPort - skipping start."
        } else {
            $wslHomeLiteral = Get-WslCommandLiteral $vllmWslHome
            $wslVenvLiteral = Get-WslCommandLiteral $vllmWslVenv
            $wslModelLiteral = Get-WslCommandLiteral $vllmWslModel
            $wslHostLiteral = Get-WslCommandLiteral $vllmWslHost
            $wslExtraArgs = $vllmWslExtraArgs
            $wslCommand = "export HOME='$wslHomeLiteral'; cd '$wslHomeLiteral'; mkdir -p '$wslHomeLiteral/logs'; source '$wslVenvLiteral/bin/activate'; python -m vllm.entrypoints.openai.api_server --model '$wslModelLiteral' --trust-remote-code --mamba_ssm_cache_dtype float32 --host '$wslHostLiteral' --port $vllmWslPort $wslExtraArgs"
            if ($watchdogEnabled) {
                Start-Process powershell -ArgumentList "-NoExit", "-Command", "
                    `$host.UI.RawUI.WindowTitle='AI-Models - vllm_wsl';
                    while (`$true) {
                        Write-Host '[watchdog] Starting vllm_wsl...';
                        & wsl -d '$vllmWslDistro' -- bash -lc ""$wslCommand"";
                        Write-Host '[watchdog] vllm_wsl exited. Restarting in 10s...';
                        Start-Sleep -Seconds 10
                    }"
            } else {
                Start-Process powershell -ArgumentList "-NoExit", "-Command", "
                    `$host.UI.RawUI.WindowTitle='AI-Models - vllm_wsl';
                    & wsl -d '$vllmWslDistro' -- bash -lc ""$wslCommand"";
                "
            }
        }
        Write-Host "Waiting for WSL vLLM on port $vllmWslPort (official NVIDIA Nemotron can take several minutes to load)..."
        $vllmUp = Wait-HttpReady -Url $vllmWslReadyUrl -TimeoutSec 900
        if ($vllmUp) {
            $vllmModels = Get-HttpJson -Url $vllmWslReadyUrl -TimeoutSec 10
            $available = @()
            if ($vllmModels -and $vllmModels.data) {
                $available = @($vllmModels.data | ForEach-Object { $_.id })
            }
            if ($available.Count -gt 0) {
                Write-Host "WSL vLLM ready. Available model(s): $($available -join ', ')"
            } else {
                Write-Warning "WSL vLLM responded but did not list any models."
            }
        } else {
            Write-Warning "WSL vLLM did not become ready within 15 minutes - check the vllm_wsl window."
        }
    } else {
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

        Write-Host "Warming local Ollama model (3 inference passes - first pass may take 5+ min)..."
        $warmPrompt = '{"action":"HOLD"}'
        $warmBody = @{
            model = $nemotronModel
            prompt = $warmPrompt
            stream = $false
            keep_alive = "1440h"
            options = @{ num_predict = 30; temperature = 0; num_ctx = 8192 }
        } | ConvertTo-Json -Compress
        for ($warmPass = 1; $warmPass -le 3; $warmPass++) {
            Write-Host "  Warmup pass $warmPass/3..."
            try {
                Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post -ContentType "application/json" -Body $warmBody -TimeoutSec 360 | Out-Null
                Write-Host "  Pass $warmPass done."
            } catch {
                Write-Warning "  Pass $warmPass failed: $_"
            }
        }
        Write-Host "Local Ollama model warm."
    }
}

Write-Host "Started persistent AI services:"
if ($needsPhi3) {
    Write-Host " - phi3_npu"
} else {
    Write-Host " - advisory_backend ($advisoryModelProvider)"
}
if ($advisoryModelProvider -eq "local_nemo" -or $nemotronStrategistProvider -eq "local") {
    if ($localLlmBackend -eq "lmstudio") {
        Write-Host " - lmstudio (local Gemma/strategy host)"
    } elseif ($localLlmBackend -eq "vllm_wsl") {
        Write-Host " - vllm_wsl (official NVIDIA Nemotron via Ubuntu/WSL)"
    } else {
        Write-Host " - ollama (local model host)"
    }
} else {
    Write-Host " - nemotron_provider ($nemotronStrategistProvider)"
}
