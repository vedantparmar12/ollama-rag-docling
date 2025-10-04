@echo off
REM ================================================================
REM RAG System Status Checker for Windows
REM ================================================================

echo ================================================================
echo 📊 RAG System Status
echo ================================================================
echo.

echo Checking service endpoints...
echo.

REM Check Frontend
echo [Frontend - Port 3000]
curl -s -f http://localhost:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Frontend: RUNNING
) else (
    echo ❌ Frontend: NOT RUNNING
)

REM Check Backend
echo [Backend - Port 8000]
curl -s -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend: RUNNING
) else (
    echo ❌ Backend: NOT RUNNING
)

REM Check RAG API
echo [RAG API - Port 8001]
curl -s -f http://localhost:8001/models >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ RAG API: RUNNING
) else (
    echo ❌ RAG API: NOT RUNNING
)

REM Check Ollama
echo [Ollama - Port 11434]
curl -s -f http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Ollama: RUNNING
) else (
    echo ❌ Ollama: NOT RUNNING
)

echo.
echo ================================================================
echo Running Processes:
echo ================================================================
echo.

echo Node.js processes:
tasklist | findstr /I "node.exe" 2>nul
if %errorlevel% neq 0 echo   None

echo.
echo Python processes:
tasklist | findstr /I "python.exe" 2>nul
if %errorlevel% neq 0 echo   None

echo.
echo Ollama processes:
tasklist | findstr /I "ollama.exe" 2>nul
if %errorlevel% neq 0 echo   None

echo.
echo ================================================================
pause
