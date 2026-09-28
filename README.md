# 🥑 Edge-Vision : Zero-Touch AI Food Freshness & Waste Analytics Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://ultralytics.com)
[![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-blue.svg)](https://github.com/JaidedAI/EasyOCR)
[![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-8E75B2.svg?logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Edge-Vision** is a zero-touch AI food freshness and waste analytics platform. It passively ingests produce shelf images from an **ESP32-CAM**, processes them through a multi-stage computer vision pipeline (**YOLOv8** produce detection, **EasyOCR** expiry date extraction, and a **PyTorch visual decay regression model**), and serves actionable zero-waste recipe recommendations through an ambient web dashboard and notification system.

---

## 🏗️ 4-Layer System Architecture

```text
[ ESP32-CAM Microcontroller / OV2640 Sensor ]
                     │
                     │  HTTP POST multipart/form-data (Scheduled Stream)
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION LAYER                                                     │
│    • Route: POST /api/v1/ingest/frame                                  │
│    • Asynchronous Frame Buffer & Telemetry (Temp/Humidity)            │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. PROCESSING & COMPUTER VISION LAYER                                  │
│    • YOLOv8 : Produce localization & categorization (Tomato, Banana)   │
│    • EasyOCR : Packaged good "Best Before" / "EXP" date extraction     │
│    • PyTorch Decay Regressor : Color/texture necrosis to Freshness %   │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. INTELLIGENCE LAYER (ZERO-WASTE RECIPE ENGINE)                       │
│    • Triggers when Freshness < 30% or Estimated Days Remaining <= 2    │
│    • Google Gemini 2.5 Flash + Fallback Structured JSON Validator      │
│    • 3 Tailored Chef-Crafted Recipes with Checklists & Step-by-Step    │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. INTERFACE LAYER & AMBIENT NOTIFICATIONS                             │
│    • Ambient Web Dashboard (Tailwind CSS, Chart.js, Lucide)           │
│    • Live Visual Inventory Grid with Detection Badges                  │
│    • Visual Decay Trajectory Curves & Time-Lapse Progress Bars         │
│    • Zero-Waste Recipe Drawer with Interactive Checklists              │
│    • Household Impact Analytics: -30% Waste, $ Savings, CO2e           │
│    • Ambient Alerts & WhatsApp Webhook Dispatch                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.12 64-bit)
- Windows / macOS / Linux

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Goku-2307/FreshRoute-AI.git
cd FreshRoute-AI

# Create virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Platform Server
Double-click `run_server.bat` or run:

```bash
.\.venv\Scripts\python -m uvicorn edge_vision.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
👉 **`http://localhost:8000`**

### 4. How to Turn Off / Stop the Server
To shut down Edge-Vision:
- In the terminal where the server is running, press **`Ctrl + C`**.
- Or simply close the Command Prompt / PowerShell window running `run_server.bat`.

---

## 📹 Live Camera Feed & Automatic Produce Detection

You can point your laptop webcam or USB camera at fruits and vegetables to automatically detect and monitor them in real-time.

### Method A: Directly from the Web Dashboard (Recommended)
1. Open **`http://localhost:8000`** in your browser.
2. Click the **"Live Camera & Auto-Detection"** tab.
3. Click **"Start Live Camera"** and allow camera permissions in your browser.
4. Click **"Start Auto-Detect"** (select interval: 1.5s, 2.5s, or 4.0s).
5. Point your camera at fruits and vegetables (e.g. tomatoes, bananas, apples). The system will continuously capture frames, run YOLOv8 object detection + PyTorch decay regression, and update your inventory in real time!
6. Click **"Stop Live Camera"** when done.

### Method B: Standalone OpenCV Streamer (Terminal / Script)
To stream your webcam to Edge-Vision via OpenCV:
- Double-click [`run_live_webcam.bat`](file:///c:/Users/adith/Documents/antigravity/serene-nobel/run_live_webcam.bat)
- Or run in terminal:
  ```powershell
  .\.venv\Scripts\python mock_esp32_cam.py --webcam 0 --interval 2
  ```
  *(Press `Q` or `ESC` in the camera preview window to quit)*

---

## 📡 Simulating ESP32-CAM Ingestion

To test the end-to-end ingestion pipeline with static sample captures:

```bash
# Send a single frame
.\.venv\Scripts\python mock_esp32_cam.py --image sample_images/fresh_tomato.jpg

# Or stream continuously every 6 seconds
.\.venv\Scripts\python mock_esp32_cam.py --loop --interval 6
```

Or double-click [`run_mock_cam.bat`](file:///c:/Users/adith/Documents/antigravity/serene-nobel/run_mock_cam.bat).

---

## 📷 Physical ESP32-CAM Setup

1. Open `firmware/esp32_cam_firmware.ino` in Arduino IDE.
2. Select Board: **AI Thinker ESP32-CAM**.
3. Configure your WiFi credentials:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* serverName = "http://<YOUR_PC_LOCAL_IP>:8000/api/v1/ingest/frame";
   ```
4. Flash the ESP32-CAM. The camera will automatically capture shelf frames every 60 seconds and transmit them to Edge-Vision.

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/ingest/frame` | ESP32-CAM scheduled frame capture ingestion |
| `GET` | `/api/v1/inventory` | List monitored produce items (`status=CRITICAL,FRESH`) |
| `GET` | `/api/v1/inventory/{item_id}` | Retrieve single item details & decay history |
| `DELETE` | `/api/v1/inventory/{item_id}` | Mark item as cooked / consumed / rescued |
| `GET` | `/api/v1/recipes/expiring` | Trigger zero-waste recipe engine for expiring items |
| `GET` | `/api/v1/decay/trajectories` | Visual decay curves and time-lapse progress data |
| `GET` | `/api/v1/analytics` | Household impact metrics (-30% waste, savings, CO2) |
| `POST` | `/api/v1/alerts/webhook` | Test or trigger ambient WhatsApp notification |
| `GET` | `/health` | Platform health status |

---

## 🧪 Running Automated Tests

Run the full integration test suite:

```bash
.\.venv\Scripts\python -m unittest discover -s tests -v
```

All 8 tests verify:
- Server health & database models
- YOLOv8 produce detector & HUD annotation
- EasyOCR date parsing & days remaining calculations
- PyTorch visual decay regression model
- LLM zero-waste recipe engine & fallback JSON schema validation
- End-to-end camera frame ingestion route (`POST /api/v1/ingest/frame`)

---

## 📄 License
MIT License. Crafted for zero-touch food freshness monitoring and household food waste elimination.
