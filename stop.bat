@echo off
REM ================================================================
REM RAG System Stopper for Windows
REM ================================================================
REM This script stops all RAG system components
REM ================================================================

echo ================================================================
echo 🛑 Stopping RAG System
echo ================================================================
echo.

echo Stopping Node.js (Frontend)...
taskkill /F /IM node.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Frontend stopped
) else (
    echo ℹ️  No frontend process found
)

echo.
echo Stopping Python servers (Backend + RAG API)...
taskkill /F /IM python.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend and RAG API stopped
) else (
    echo ℹ️  No Python processes found
)

echo.
echo ================================================================
echo ✅ All services stopped!
echo ================================================================
echo.
echo Note: Ollama server is still running (not stopped by this script)
echo If you want to stop Ollama, close it manually or run: taskkill /F /IM ollama.exe
echo.
pause
