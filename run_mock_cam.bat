@echo off
title ESP32-CAM Mock Stream Simulator
echo =====================================================================
echo  EDGE-VISION : ESP32-CAM INGESTION STREAM SIMULATOR
echo =====================================================================

if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe mock_esp32_cam.py --loop --interval 6
) else (
    py -3.12 mock_esp32_cam.py --loop --interval 6
)
pause
