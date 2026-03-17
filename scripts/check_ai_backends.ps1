function Get-EnvValue {
    param(
        [string]$Key,
        [string]$Default = ""
    )
    $envFile = Join-Path $PSScriptRoot "..\.env"
    if (-not (Test-Path $envFile)) {
        return $Default
    }
    $line = Get-Content $envFile | Where-Object { $_ -match "^\s*$Key=" } | Select-Object -First 1
    if (-not $line) {
        return $Default
    }
    return ($line -split "=", 2)[1].Trim()
}

$nemotronProvider = (Get-EnvValue "NEMOTRON_PROVIDER" "local").ToLower()
$nemotronBaseUrl = Get-EnvValue "NEMOTRON_BASE_URL" "http://127.0.0.1:8081"
$ports = @(
    @{ Name = "Phi-3"; Url = "http://127.0.0.1:8084/health" },
    @{ Name = "Atlas"; Url = "http://127.0.0.1:8083/health" }
)

if ($nemotronProvider -eq "local") {
    $ports += @{ Name = "Nemotron"; Url = "$nemotronBaseUrl/v1/models"; Type = "openai_models" }
}

$results = @()

if ($nemotronProvider -ne "local") {
    $results += [pscustomobject]@{
        Service = "Nemotron"
        Url = "NVIDIA API"
        Status = "configured"
        Device = "cloud"
    }
}

foreach ($svc in $ports) {
    try {
        $resp = Invoke-RestMethod -Uri $svc.Url -Method Get -TimeoutSec 5
        $status = "ok"
        if ($svc.Type -eq "openai_models") {
            $status = if ($resp.data) { "ok" } else { "unexpected" }
        } elseif ($resp.status) {
            $status = ($resp.status | Out-String).Trim()
        }
        $results += [pscustomobject]@{
            Service = $svc.Name
            Url = $svc.Url
            Status = $status
            Device = if ($svc.Type -eq "openai_models") { "local" } else { ($resp.device | Out-String).Trim() }
        }
    } catch {
        $results += [pscustomobject]@{
            Service = $svc.Name
            Url = $svc.Url
            Status = "unreachable"
            Device = ""
        }
    }
}

$results | Format-Table -AutoSize
