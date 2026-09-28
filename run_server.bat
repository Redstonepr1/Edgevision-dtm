@echo off
title Edge-Vision Platform Server
echo =====================================================================
echo  EDGE-VISION : AI FOOD FRESHNESS & ZERO-WASTE ANALYTICS PLATFORM
echo =====================================================================
echo Starting Edge-Vision server on http://localhost:8000 ...

if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe -m uvicorn edge_vision.main:app --host 0.0.0.0 --port 8000 --reload
) else (
    py -3.12 -m uvicorn edge_vision.main:app --host 0.0.0.0 --port 8000 --reload
)
pause
