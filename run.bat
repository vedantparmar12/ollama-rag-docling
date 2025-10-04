@echo off
REM ================================================================
REM RAG System Launcher for Windows
REM ================================================================
REM This script starts all RAG system components:
REM - RAG API Server (port 8001)
REM - Backend Server (port 8000)
REM - Frontend Server (port 3000)
REM ================================================================

setlocal enabledelayedexpansion

echo ================================================================
echo 🚀 Starting RAG System - Ollama Edition
echo ================================================================
echo.

REM Check if Ollama is running
echo [1/4] Checking Ollama status...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Ollama is not running!
    echo Please start Ollama first and ensure it's running on port 11434
    echo.
    echo You can start Ollama by running: ollama serve
    pause
    exit /b 1
) else (
    echo ✅ Ollama is running
)

echo.
echo [2/4] Checking required models...
ollama list | findstr /C:"llama3.2" >nul
if %errorlevel% neq 0 (
    echo ⚠️  Warning: llama3.2 not found. Pulling model...
    ollama pull llama3.2
)

ollama list | findstr /C:"smollm2" >nul
if %errorlevel% neq 0 (
    echo ⚠️  Warning: smollm2 not found. Pulling model...
    ollama pull smollm2
)

ollama list | findstr /C:"nomic-embed-text" >nul
if %errorlevel% neq 0 (
    echo ⚠️  Warning: nomic-embed-text not found. Pulling model...
    ollama pull nomic-embed-text
)

echo ✅ All required models are available

REM Create necessary directories
echo.
echo [3/4] Setting up directories...
if not exist "lancedb" mkdir lancedb
if not exist "shared_uploads" mkdir shared_uploads
if not exist "logs" mkdir logs
if not exist "index_store" mkdir index_store
if not exist "index_store\overviews" mkdir index_store\overviews
if not exist "index_store\bm25" mkdir index_store\bm25
if not exist "index_store\graph" mkdir index_store\graph
echo ✅ Directories created

REM Get current directory for PYTHONPATH
set CURRENT_DIR=%CD%

REM Start services in new windows
echo.
echo [4/4] Starting services...

REM Start RAG API Server
echo 🔧 Starting RAG API Server on port 8001...
start "RAG API Server" cmd /k "set PYTHONPATH=%CURRENT_DIR% && python rag_system\api_server.py"
timeout /t 3 /nobreak >nul

REM Start Backend Server
echo 🔧 Starting Backend Server on port 8000...
start "Backend Server" cmd /k "set PYTHONPATH=%CURRENT_DIR% && cd backend && python server.py"
timeout /t 3 /nobreak >nul

REM Start Frontend Server
echo 🔧 Starting Frontend Server on port 3000...
start "Frontend Server" cmd /k "npm run dev"
timeout /t 3 /nobreak >nul

echo.
echo ================================================================
echo ✅ RAG System Started Successfully!
echo ================================================================
echo.
echo 🌐 Access your application at:
echo    Frontend:   http://localhost:3000
echo    Backend:    http://localhost:8000
echo    RAG API:    http://localhost:8001
echo    Ollama:     http://localhost:11434
echo.
echo 📊 Models in use:
echo    Generation: llama3.2:latest
echo    Enrichment: smollm2:latest
echo    Embedding:  nomic-embed-text:latest
echo.
echo 📝 Three new windows have opened for each service.
echo    Close those windows to stop the services.
echo.
echo 🎯 Next steps:
echo    1. Open http://localhost:3000 in your browser
echo    2. Create a new chat session
echo    3. Upload some PDF documents
echo    4. Start asking questions!
echo.
echo Press any key to open the frontend in your browser...
pause >nul

REM Open browser
start http://localhost:3000

echo.
echo System is running. Close this window when done.
echo ================================================================

endlocal
