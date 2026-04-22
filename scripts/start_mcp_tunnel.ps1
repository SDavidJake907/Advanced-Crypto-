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

function Wait-HttpReady {
    param(
        [string]$Url,
        [int]$TimeoutSec = 60
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5 -Headers @{ "Accept" = "text/event-stream" } -ErrorAction SilentlyContinue
            if ($null -ne $resp) {
                return $true
            }
        } catch {
            if ($null -ne $_.Exception.Response) {
                return $true
            }
            Write-Host "Wait-HttpReady attempt failed: $($_.Exception.Message)"
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

$mcpHost = Get-EnvValue "MCP_HOST" "127.0.0.1"
$mcpPort = Get-EnvValue "MCP_PORT" "8765"
$mcpPublicEnabled = (Get-EnvValue "MCP_PUBLIC_ENABLED" "false").ToLower() -eq "true"
$cloudflaredExe = Get-EnvValue "CLOUDFLARED_EXE" "cloudflared"
$mcpPublicHostname = Get-EnvValue "MCP_PUBLIC_HOSTNAME" ""
$mcpTunnelName = Get-EnvValue "MCP_TUNNEL_NAME" ""
$mcpLocalUrl = "http://${mcpHost}:${mcpPort}"

Write-Host "Checking MCP server at $mcpLocalUrl ..."

if (-not $mcpPublicEnabled) {
    Write-Host "MCP public tunnel disabled. Set MCP_PUBLIC_ENABLED=true to run cloudflared."
    exit 0
}

if (-not (Wait-HttpReady -Url "$mcpLocalUrl/" -TimeoutSec 30)) {
    Write-Error "MCP server is not reachable at $mcpLocalUrl"
    exit 1
}

$cloudflaredCmd = Get-Command $cloudflaredExe -ErrorAction SilentlyContinue
if (-not $cloudflaredCmd) {
    Write-Error "cloudflared executable not found. Set CLOUDFLARED_EXE in .env or install cloudflared."
    exit 1
}

if ($mcpPublicHostname -and $mcpTunnelName) {
    Write-Host "Starting named Cloudflare tunnel '$mcpTunnelName' for $mcpPublicHostname -> $mcpLocalUrl"
    & $cloudflaredCmd.Source tunnel --url $mcpLocalUrl --hostname $mcpPublicHostname run $mcpTunnelName
} else {
    Write-Host "Starting quick Cloudflare tunnel for $mcpLocalUrl"
    & $cloudflaredCmd.Source tunnel --url $mcpLocalUrl
}
