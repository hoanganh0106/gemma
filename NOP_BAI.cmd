@echo off
chcp 65001 >nul
title MOA - Nop bai tu dong
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_moa_submit.ps1"
echo.
pause
