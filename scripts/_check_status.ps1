# KrakenSK Status Check
$root = "C:\Users\kitti\Desktop\KrakenSK"
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

$localLlmBackend = (Get-EnvValue "LOCAL_LLM_BACKEND" "ollama").ToLower()
$localLlmBaseUrl = Get-EnvValue "ADVISORY_LOCAL_BASE_URL" (Get-EnvValue "NEMOTRON_BASE_URL" "http://127.0.0.1:11434")
$localLlmModel = Get-EnvValue "ADVISORY_LOCAL_MODEL" (Get-EnvValue "NEMOTRON_MODEL" "")

Write-Host "`n=== SERVICES ===" -ForegroundColor Cyan

if ($localLlmBackend -eq "lmstudio") {
    try {
        $modelsUrl = if ($localLlmBaseUrl.TrimEnd('/').EndsWith('/v1')) { "$($localLlmBaseUrl.TrimEnd('/'))/models" } else { "$($localLlmBaseUrl.TrimEnd('/'))/v1/models" }
        $r = Invoke-RestMethod $modelsUrl -TimeoutSec 5
        $modelName = if ($localLlmModel) { $localLlmModel } else { "available" }
        Write-Host "  LM Studio:   OK - $modelName" -ForegroundColor Green
    } catch {
        Write-Host "  LM Studio:   OFFLINE" -ForegroundColor Red
    }
} else {
    try { $r = Invoke-RestMethod 'http://127.0.0.1:11434/api/ps' -TimeoutSec 5
        if ($r.models.Count -gt 0) { Write-Host "  Ollama:      OK - $($r.models[0].name) ($([int]($r.models[0].size_vram/1GB))GB VRAM)" -ForegroundColor Green }
        else { Write-Host "  Ollama:      RUNNING but no model loaded" -ForegroundColor Yellow }
    } catch { Write-Host "  Ollama:      OFFLINE" -ForegroundColor Red }
}

# Phi3 NPU
try { $h = Invoke-RestMethod 'http://127.0.0.1:8084/health' -TimeoutSec 5
    Write-Host "  Phi3 NPU:    OK - status=$($h.status)" -ForegroundColor Green
} catch { Write-Host "  Phi3 NPU:    OFFLINE / still loading" -ForegroundColor Yellow }

# Operator UI
try { Invoke-WebRequest 'http://127.0.0.1:8780' -UseBasicParsing -TimeoutSec 5 | Out-Null
    Write-Host "  Operator UI: OK (http://127.0.0.1:8780)" -ForegroundColor Green
} catch { Write-Host "  Operator UI: OFFLINE" -ForegroundColor Red }

Write-Host "`n=== PROCESSES ===" -ForegroundColor Cyan
$py = (Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count
$ol = (Get-Process ollama -ErrorAction SilentlyContinue | Measure-Object).Count
$lm = (Get-Process 'LM Studio' -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "  Python:  $py process(es)"
Write-Host "  Ollama:  $ol process(es)"
Write-Host "  LM Studio: $lm process(es)"

Write-Host "`n=== LOG ACTIVITY ===" -ForegroundColor Cyan
$logs = @{
    "Decision"  = "$root\logs\decision_debug.jsonl"
    "Warmup"    = "$root\logs\warmup.jsonl"
    "Collector" = "$root\logs\collector_telemetry.json"
    "Nemotron"  = "$root\logs\nemotron_debug.jsonl"
}
foreach ($name in $logs.Keys) {
    $path = $logs[$name]
    if (Test-Path $path) {
        $age = [int]((Get-Date) - (Get-Item $path).LastWriteTime).TotalSeconds
        $status = if ($age -lt 120) { "ACTIVE" } elseif ($age -lt 600) { "SLOW" } else { "STALE" }
        $color = if ($age -lt 120) { "Green" } elseif ($age -lt 600) { "Yellow" } else { "Red" }
        Write-Host "  $name`: $status (${age}s ago)" -ForegroundColor $color
    } else {
        Write-Host "  $name`: not found" -ForegroundColor Yellow
    }
}

# Last decision
$dlog = "$root\logs\decision_debug.jsonl"
if (Test-Path $dlog) {
    Write-Host "`n=== LAST 3 DECISIONS ===" -ForegroundColor Cyan
    Get-Content $dlog -Tail 10 | ForEach-Object {
        try {
            $d = $_ | ConvertFrom-Json
            if ($d.symbol -and $d.ts) {
                $nem = $d.nemotron
                $action = if ($nem) { $nem.action } else { "gate" }
                Write-Host "  $($d.ts.Substring(0,16))  $($d.symbol.PadRight(14)) $action"
            }
        } catch {}
    } | Select-Object -Last 3
}

# Collector state
$tel = "$root\logs\collector_telemetry.json"
if (Test-Path $tel) {
    $t = Get-Content $tel | ConvertFrom-Json
    Write-Host "`n=== COLLECTOR ===" -ForegroundColor Cyan
    Write-Host "  Symbols tracked: $($t.symbol_count)"
}
