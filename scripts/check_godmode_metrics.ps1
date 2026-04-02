param(
    [string]$Cutoff = ""
)

$root = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $PSScriptRoot "check_godmode_metrics.py"

if (-not (Test-Path $scriptPath)) {
    Write-Error "Missing script: $scriptPath"
    exit 1
}

$args = @($scriptPath)
if ($Cutoff) {
    $args += @("--cutoff", $Cutoff)
}

python @args
