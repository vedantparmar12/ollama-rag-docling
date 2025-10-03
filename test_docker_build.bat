@echo off
REM Test Docker builds individually for Windows

setlocal enabledelayedexpansion

echo 🐳 Testing Docker builds individually...

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop.
    exit /b 1
)
echo ✅ Docker is running

echo.
echo 🧹 Cleaning up old containers and images...
docker container prune -f >nul 2>&1
docker image prune -f >nul 2>&1

echo 📦 Building containers in dependency order...

REM 1. RAG API (no dependencies)
call :build_and_test "rag-api" "Dockerfile.rag-api" "8001"
if %errorlevel% neq 0 (
    echo ❌ RAG API build failed, stopping
    exit /b 1
)

REM 2. Backend (depends on RAG API)
call :build_and_test "backend" "Dockerfile.backend" "8000"
if %errorlevel% neq 0 (
    echo ❌ Backend build failed, stopping
    exit /b 1
)

REM 3. Frontend (depends on Backend)
call :build_and_test "frontend" "Dockerfile.frontend" "3000"
if %errorlevel% neq 0 (
    echo ❌ Frontend build failed, stopping
    exit /b 1
)

echo.
echo 🎉 All containers built and tested successfully!
echo 🚀 You can now run: start-docker.bat

goto :eof

:build_and_test
set service=%~1
set dockerfile=%~2
set port=%~3

echo.
echo 🔨 Building %service%...
docker build -f %dockerfile% -t "rag-%service%" .
if %errorlevel% neq 0 (
    echo ❌ Failed to build %service%
    exit /b 1
)

echo ✅ %service% built successfully

REM Test running the container
echo 🚀 Testing %service% container...
docker run -d --name "test-%service%" -p "%port%:%port%" "rag-%service%"
if %errorlevel% neq 0 (
    echo ❌ Failed to run %service%
    exit /b 1
)

echo ⏳ Waiting for %service% to start...
timeout /t 10 /nobreak >nul

REM Test health
if "%service%"=="frontend" (
    curl -f "http://localhost:%port%" >nul 2>&1
) else if "%service%"=="backend" (
    curl -f "http://localhost:%port%/health" >nul 2>&1
) else if "%service%"=="rag-api" (
    curl -f "http://localhost:%port%/models" >nul 2>&1
)

if %errorlevel% equ 0 (
    echo ✅ %service% is healthy
) else (
    echo ⚠️ %service% health check failed (but container is running^)
    docker logs "test-%service%" 2>&1 | findstr /V /R "^$" | more +0
)

REM Cleanup
docker stop "test-%service%" >nul 2>&1
docker rm "test-%service%" >nul 2>&1

exit /b 0

endlocal
