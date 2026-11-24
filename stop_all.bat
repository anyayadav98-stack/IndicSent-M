@echo off
echo Stopping IndicSent-M Backend (port 8000) and Frontend (ports 5173/5174)...

powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetTCPConnection -LocalPort 8000, 5173, 5174 -ErrorAction SilentlyContinue | ForEach-Object { try { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue; Write-Host ('Killed PID ' + $_.OwningProcess + ' on port ' + $_.LocalPort) } catch {} }"

echo All IndicSent-M processes on ports 8000, 5173, 5174 stopped.
