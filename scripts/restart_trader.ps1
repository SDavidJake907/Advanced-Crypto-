$root = "C:\Users\kitti\Desktop\KrakenSK"
$python = Join-Path $root ".venv\Scripts\python.exe"

# Kill existing trader
Get-Process python* -ErrorAction SilentlyContinue | Where-Object {
    $_.MainWindowTitle -like "*trader*"
} | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", "`$host.UI.RawUI.WindowTitle='KrakenSK - trader'; cd '$root'; & '$python' -m apps.trader.main"
)

Write-Host "Trader restarted."
