@echo off
REM LocalGPT Docker Startup Script for Windows
REM This script provides easy options for running LocalGPT in Docker

setlocal enabledelayedexpansion

echo 🐳 LocalGPT Docker Deployment
echo ============================
echo.

REM Parse command line argument
set "option=%~1"
if "%option%"=="" set "option=local"

if "%option%"=="local" goto start_local
if "%option%"=="container" goto start_container
if "%option%"=="stop" goto stop_containers
if "%option%"=="logs" goto show_logs
if "%option%"=="status" goto show_status
if "%option%"=="help" goto show_usage
if "%option%"=="-h" goto show_usage
if "%option%"=="--help" goto show_usage

echo ❌ Unknown option: %option%
echo.
goto show_usage

:check_local_ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Local Ollama detected on port 11434
    exit /b 0
) else (
    echo ❌ No local Ollama detected on port 11434
    exit /b 1
)

:start_local
call :check_local_ollama
if %errorlevel% equ 0 (
    echo 🚀 Starting LocalGPT containers (using local Ollama^)...
    echo 📝 Note: Make sure your local Ollama is running on port 11434

    docker compose --env-file docker.env up --build -d

    echo.
    echo 🎉 LocalGPT is starting up!
    echo 📱 Frontend: http://localhost:3000
    echo 🔧 Backend API: http://localhost:8000
    echo 🧠 RAG API: http://localhost:8001
    echo 🤖 Ollama: http://localhost:11434 (local^)
    echo.
    echo 📊 Check container status: docker compose ps
    echo 📝 View logs: docker compose logs -f
    echo 🛑 Stop services: docker compose down
) else (
    echo.
    echo ⚠️  No local Ollama detected. Options:
    echo 1. Start local Ollama: 'ollama serve'
    echo 2. Use containerized Ollama: '%~nx0 container'
    echo.
    set /p "REPLY=Start with containerized Ollama instead? (y/N): "
    if /i "!REPLY!"=="y" (
        goto start_container
    ) else (
        echo ❌ Cancelled. Please start local Ollama or use '%~nx0 container'
        exit /b 1
    )
)
goto :eof

:start_container
echo 🚀 Starting LocalGPT containers (including Ollama container^)...

set OLLAMA_HOST=http://ollama:11434

docker compose --profile with-ollama up --build -d

echo.
echo 🎉 LocalGPT is starting up!
echo 📱 Frontend: http://localhost:3000
echo 🔧 Backend API: http://localhost:8000
echo 🧠 RAG API: http://localhost:8001
echo 🤖 Ollama: http://localhost:11434 (containerized^)
echo.
echo ⏳ Note: First startup may take longer as Ollama container initializes
echo 📊 Check container status: docker compose --profile with-ollama ps
echo 📝 View logs: docker compose --profile with-ollama logs -f
echo 🛑 Stop services: docker compose --profile with-ollama down
goto :eof

:stop_containers
echo 🛑 Stopping LocalGPT containers...
docker compose down
docker compose --profile with-ollama down 2>nul
echo ✅ All containers stopped
goto :eof

:show_logs
echo 📝 Showing container logs (Ctrl+C to exit^)...
docker compose ps | findstr "rag-ollama" >nul 2>&1
if %errorlevel% equ 0 (
    docker compose --profile with-ollama logs -f
) else (
    docker compose logs -f
)
goto :eof

:show_status
echo 📊 Container Status:
docker compose ps
echo.
echo 🐳 All Docker containers:
docker ps | findstr /C:"rag-" /C:"CONTAINER"
if %errorlevel% neq 0 echo No LocalGPT containers running
goto :eof

:show_usage
echo Usage: %~nx0 [option]
echo.
echo Options:
echo   local     - Use local Ollama instance (default^)
echo   container - Use containerized Ollama
echo   stop      - Stop all containers
echo   logs      - Show container logs
echo   status    - Show container status
echo   help      - Show this help message
echo.
echo Examples:
echo   %~nx0 local      # Use local Ollama (recommended^)
echo   %~nx0 container  # Use containerized Ollama
echo   %~nx0 stop       # Stop all services
goto :eof

endlocal
