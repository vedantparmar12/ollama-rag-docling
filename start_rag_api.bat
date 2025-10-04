@echo off
echo Starting RAG API Server on port 8001...
set PYTHONPATH=%CD%
python rag_system\api_server.py
pause
