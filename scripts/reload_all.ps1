$root = "C:\Users\kitti\Desktop\KrakenSK"
$signal = Join-Path $root "logs\reload.signal"

New-Item -ItemType Directory -Force -Path (Split-Path $signal) | Out-Null
Set-Content -Path $signal -Value ([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())

Write-Host "KrakenSK reload signal written:"
Write-Host " - $signal"
Write-Host "Trader and collector will reload symbols/runtime state on the next loop."
