@echo off
title Edge-Vision Live Webcam Streamer
echo =====================================================================
echo  EDGE-VISION : LIVE CAMERA PRODUCE AUTO-DETECTOR
echo =====================================================================
echo Streaming live webcam frames to Edge-Vision AI backend...
echo (Press 'Q' or 'ESC' in the camera window to stop)

if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe mock_esp32_cam.py --webcam 0 --interval 2
) else (
    py -3.12 mock_esp32_cam.py --webcam 0 --interval 2
)
pause
