@echo off
echo Starting Backend Server on port 8000...
set PYTHONPATH=%CD%
cd backend
python server.py
pause
