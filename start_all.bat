@echo off
echo ================================================================
echo Starting RAG System - All Services
echo ================================================================
echo.
echo Opening 3 windows for:
echo   1. RAG API Server (port 8001)
echo   2. Backend Server (port 8000)
echo   3. Frontend Server (port 3000)
echo.

start "RAG API Server" cmd /k start_rag_api.bat
ping 127.0.0.1 -n 3 >nul

start "Backend Server" cmd /k start_backend.bat
ping 127.0.0.1 -n 3 >nul

start "Frontend Server" cmd /k start_frontend.bat
ping 127.0.0.1 -n 3 >nul

echo.
echo ================================================================
echo All services started!
echo ================================================================
echo.
echo Access the app at: http://localhost:3000
echo.
echo To stop: Close the 3 terminal windows or run stop.bat
echo.
echo Opening browser in 5 seconds...
ping 127.0.0.1 -n 6 >nul
start http://localhost:3000
