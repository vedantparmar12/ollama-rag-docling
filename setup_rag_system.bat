@echo off
REM setup_rag_system.bat - Complete RAG System Setup Script for Windows
REM This script handles Docker installation verification, system setup, and initial configuration

setlocal enabledelayedexpansion

echo ================================================================
echo 🚀 RAG System Complete Setup Script (Windows)
echo ================================================================
echo.

REM Step 1: System Requirements Check
echo [Step 1] Checking system requirements...

REM Check if Docker is installed
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed. Please install Docker Desktop for Windows from:
    echo https://www.docker.com/products/docker-desktop/
    exit /b 1
) else (
    echo INFO: Docker is installed
    docker --version
)

REM Check if Docker is running
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop.
    exit /b 1
) else (
    echo INFO: Docker is running
)

REM Check if Docker Compose is available
docker compose version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker Compose is not available
    exit /b 1
) else (
    echo INFO: Docker Compose is available
    docker compose version
)

REM Step 2: Setup RAG System
echo.
echo [Step 2] Setting up RAG System...

REM Create project directory structure
echo INFO: Creating directory structure...
if not exist "lancedb" mkdir lancedb
if not exist "shared_uploads" mkdir shared_uploads
if not exist "logs" mkdir logs
if not exist "ollama_data" mkdir ollama_data
if not exist "index_store" mkdir index_store
if not exist "index_store\overviews" mkdir index_store\overviews
if not exist "index_store\bm25" mkdir index_store\bm25
if not exist "index_store\graph" mkdir index_store\graph
if not exist "backups" mkdir backups

REM Create environment file
if not exist ".env" (
    echo INFO: Creating environment configuration...
    (
        echo # System Configuration
        echo NODE_ENV=production
        echo LOG_LEVEL=info
        echo DEBUG=false
        echo.
        echo # Service URLs
        echo FRONTEND_URL=http://localhost:3000
        echo BACKEND_URL=http://localhost:8000
        echo RAG_API_URL=http://localhost:8001
        echo OLLAMA_URL=http://localhost:11434
        echo.
        echo # Database Configuration
        echo DATABASE_PATH=./backend/chat_data.db
        echo LANCEDB_PATH=./lancedb
        echo UPLOADS_PATH=./shared_uploads
        echo INDEX_STORE_PATH=./index_store
        echo.
        echo # Model Configuration
        echo DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
        echo DEFAULT_GENERATION_MODEL=qwen3:8b
        echo DEFAULT_RERANKER_MODEL=answerdotai/answerai-colbert-small-v1
        echo DEFAULT_ENRICHMENT_MODEL=qwen3:0.6b
        echo.
        echo # Performance Configuration
        echo MAX_CONCURRENT_REQUESTS=5
        echo REQUEST_TIMEOUT=300
        echo EMBEDDING_BATCH_SIZE=32
        echo MAX_CONTEXT_LENGTH=4096
        echo.
        echo # Security Configuration
        echo CORS_ORIGINS=http://localhost:3000
        echo API_KEY_REQUIRED=false
        echo RATE_LIMIT_REQUESTS=100
        echo RATE_LIMIT_WINDOW=60
        echo.
        echo # Storage Configuration
        echo MAX_FILE_SIZE=50MB
        echo MAX_UPLOAD_FILES=10
        echo CLEANUP_INTERVAL=3600
        echo BACKUP_RETENTION_DAYS=30
    ) > .env
    echo INFO: Environment file created: .env
) else (
    echo INFO: Environment file already exists: .env
)

REM Step 3: Build and Start Services
echo.
echo [Step 3] Building and starting services...
echo INFO: Building Docker containers (this may take 10-15 minutes)...
docker compose build --no-cache

echo INFO: Starting services...
docker compose up -d

REM Wait for services to start
echo INFO: Waiting for services to initialize...
timeout /t 30 /nobreak >nul

REM Check service status
echo INFO: Checking service status...
docker compose ps

REM Step 4: Install AI Models
echo.
echo [Step 4] Installing AI models...

REM Wait for Ollama to be ready
echo INFO: Waiting for Ollama to be ready...
set max_attempts=30
set attempt=0

:ollama_wait_loop
docker compose exec ollama ollama list >nul 2>&1
if %errorlevel% equ 0 goto ollama_ready
set /a attempt+=1
if %attempt% geq %max_attempts% (
    echo ERROR: Ollama failed to start after %max_attempts% attempts
    exit /b 1
)
echo INFO: Waiting for Ollama... (attempt %attempt%/%max_attempts%)
timeout /t 10 /nobreak >nul
goto ollama_wait_loop

:ollama_ready
echo INFO: Ollama is ready

REM Download Ollama models
echo INFO: Downloading required Ollama models...
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama pull qwen3:0.6b

echo INFO: Verifying model installation...
docker compose exec ollama ollama list

REM Step 5: Create Helper Scripts
echo.
echo [Step 5] Creating helper scripts...

REM Create start script
(
    echo @echo off
    echo echo Starting RAG System...
    echo docker compose up -d
    echo echo RAG System started. Access at: http://localhost:3000
) > start_rag_system.bat

REM Create stop script
(
    echo @echo off
    echo echo Stopping RAG System...
    echo docker compose down
    echo echo RAG System stopped.
) > stop_rag_system.bat

REM Create status script
(
    echo @echo off
    echo echo === RAG System Status ===
    echo docker compose ps
    echo.
    echo echo === Service Health ===
    echo curl -s -f http://localhost:3000 && echo ✅ Frontend: OK ^|^| echo ❌ Frontend: FAIL
    echo curl -s -f http://localhost:8000/health && echo ✅ Backend: OK ^|^| echo ❌ Backend: FAIL
    echo curl -s -f http://localhost:8001/models && echo ✅ RAG API: OK ^|^| echo ❌ RAG API: FAIL
    echo curl -s -f http://localhost:11434/api/tags && echo ✅ Ollama: OK ^|^| echo ❌ Ollama: FAIL
) > status_rag_system.bat

echo.
echo ================================================================
echo 🎉 RAG System Setup Complete!
echo ================================================================
echo.
echo ✅ System Status:
echo    - Frontend: http://localhost:3000
echo    - Backend API: http://localhost:8000
echo    - RAG API: http://localhost:8001
echo    - Ollama: http://localhost:11434
echo.
echo 📚 Documentation:
echo    - System Overview: Documentation\system_overview.md
echo    - Deployment Guide: Documentation\deployment_guide.md
echo    - Docker Usage: Documentation\docker_usage.md
echo    - Installation Guide: Documentation\installation_guide.md
echo.
echo 🔧 Helper Scripts:
echo    - Start system: start_rag_system.bat
echo    - Stop system: stop_rag_system.bat
echo    - Check status: status_rag_system.bat
echo.
echo 🚀 Next Steps:
echo    1. Open http://localhost:3000 in your browser
echo    2. Create a new chat session
echo    3. Upload some PDF documents
echo    4. Start asking questions about your documents!
echo.

endlocal
