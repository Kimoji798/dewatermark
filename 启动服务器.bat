@echo off
chcp 65001 >nul
title TTmark LAN Server
cd /d "%~dp0"

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js not found. Install it from https://nodejs.org first.
  echo.
  pause
  exit /b 1
)

node serve.js

echo.
echo Server stopped. You can close this window now.
pause