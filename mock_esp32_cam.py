#!/usr/bin/env python
"""
ESP32-CAM Ingestion Mock Simulator for Edge-Vision Platform.
Simulates an ESP32-CAM microcontroller streaming produce shelf frames
to POST /api/v1/ingest/frame.
"""

import os
import sys
import time
import argparse
import random
import requests
from pathlib import Path

# Force UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/api/v1/ingest/frame"
SAMPLE_DIR = Path("sample_images")


def find_sample_images():
    """Finds all available sample produce images"""
    exts = [".jpg", ".jpeg", ".png"]
    images = []
    if SAMPLE_DIR.exists():
        for p in SAMPLE_DIR.iterdir():
            if p.suffix.lower() in exts:
                images.append(p)
    return images


def send_frame(image_path: Path, endpoint: str, device_id: str = "ESP32-CAM-NODE-01"):
    """Sends a single image frame to the Edge-Vision ingestion endpoint"""
    if not image_path.exists():
        print(f"❌ Error: Image file not found at {image_path}")
        return False

    temp = round(21.5 + random.uniform(-1.5, 3.0), 1)
    humidity = round(65.0 + random.uniform(-5.0, 10.0), 1)

    print(f"\n📡 [ESP32-CAM] Preparing frame payload: {image_path.name}")
    print(f"   ├─ Device ID    : {device_id}")
    print(f"   ├─ Telemetry    : {temp}°C | {humidity}% RH")
    print(f"   └─ Target Route : {endpoint}")

    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            data = {
                "device_id": device_id,
                "temperature_c": str(temp),
                "humidity_rh": str(humidity)
            }
            start_t = time.time()
            resp = requests.post(endpoint, files=files, data=data, timeout=15)
            elapsed_ms = int((time.time() - start_t) * 1000)

        if resp.status_code == 200:
            res_json = resp.json()
            print(f"✅ [200 OK] Frame Ingested in {elapsed_ms}ms!")
            print(f"   ├─ Frame ID         : {res_json.get('frame_id')}")
            print(f"   ├─ Items Detected   : {res_json.get('detected_count')}")
            print(f"   ├─ Critical Items   : {res_json.get('critical_items_count')}")
            print(f"   ├─ Recipes Triggered: {res_json.get('recipes_triggered')}")
            
            for item in res_json.get("items", []):
                cat = item.get("category")
                conf = item.get("confidence_score")
                fresh = item.get("freshness_percentage")
                days = item.get("estimated_days_left")
                stat = item.get("status")
                ocr = item.get("ocr_expiry_date")
                ocr_str = f" | OCR Expiry: {ocr}" if ocr else ""
                print(f"   │  └─ {cat.upper()}: {fresh}% Fresh | {days}d left | Status: {stat} (Conf: {conf}%){ocr_str}")
            
            return True
        else:
            print(f"❌ [Error {resp.status_code}] {resp.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Failed: Could not reach {endpoint}")
        print("   Please ensure the Edge-Vision server is running (e.g. uvicorn edge_vision.main:app)")
        return False
    except Exception as e:
        print(f"❌ Ingestion Error: {e}")
        return False


def stream_from_webcam(cam_index: int = 0, endpoint: str = DEFAULT_ENDPOINT, interval: int = 3, device_id: str = "LIVE-WEBCAM-01", show_window: bool = True):
    """Captures live video frames from webcam and auto-detects produce at regular intervals"""
    try:
        import cv2
    except ImportError:
        print("❌ Error: opencv-python is required for webcam streaming.")
        return

    print(f"🔌 Opening hardware webcam device index {cam_index}...")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"❌ Error: Could not open camera device {cam_index}. Check permissions and connection.")
        return

    print(f"✅ Camera opened successfully! Streaming frames every {interval}s to {endpoint}")
    print("   Controls: Press 'q' or 'ESC' in the preview window to exit.\n")

    last_send = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            now = time.time()
            if now - last_send >= interval:
                ok, jpeg_buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    temp = round(21.5 + random.uniform(-1.0, 2.0), 1)
                    humidity = round(65.0 + random.uniform(-4.0, 6.0), 1)
                    try:
                        files = {"file": ("webcam_frame.jpg", jpeg_buf.tobytes(), "image/jpeg")}
                        data = {"device_id": device_id, "temperature_c": str(temp), "humidity_rh": str(humidity)}
                        resp = requests.post(endpoint, files=files, data=data, timeout=8)
                        if resp.status_code == 200:
                            res_json = resp.json()
                            items = [f"{i.get('category')} ({i.get('freshness_percentage')}% Fresh, {i.get('estimated_days_left')}d left)" for i in res_json.get("items", [])]
                            summary = ", ".join(items) if items else "No produce detected"
                            print(f"🟢 [Live Detect] {summary}")
                    except Exception as ex:
                        print(f"⚠️ Live Ingest warning: {ex}")
                last_send = now

            if show_window:
                hud = frame.copy()
                cv2.putText(hud, "EDGE-VISION LIVE CAM - Auto-Detect Active (Press Q to quit)", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)
                cv2.imshow("Edge-Vision Live Camera Feed", hud)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n🛑 Camera released.")


def main():
    parser = argparse.ArgumentParser(description="Edge-Vision ESP32-CAM Ingestion Simulator & Live Camera Feed")
    parser.add_argument("--image", type=str, default=None, help="Path to specific image file to send")
    parser.add_argument("--webcam", type=int, default=None, help="Camera device index (e.g. 0 for built-in laptop webcam) for live auto-detection")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="Ingestion API endpoint URL")
    parser.add_argument("--loop", action="store_true", help="Continuously cycle through sample images on an interval")
    parser.add_argument("--interval", type=int, default=3, help="Loop/auto-detect interval in seconds (default: 3s)")
    parser.add_argument("--device", type=str, default="ESP32-CAM-SHELF-01", help="Device identifier string")

    args = parser.parse_args()

    print("=" * 65)
    print(" 📸 EDGE-VISION : LIVE CAMERA INGESTION & AUTO-DETECTOR")
    print("=" * 65)

    if args.webcam is not None:
        stream_from_webcam(cam_index=args.webcam, endpoint=args.endpoint, interval=args.interval, device_id=args.device)
        return

    if args.image:
        target = Path(args.image)
        send_frame(target, args.endpoint, args.device)
        return

    sample_images = find_sample_images()
    if not sample_images:
        print("❌ No sample images found in sample_images/ directory.")
        sys.exit(1)

    print(f"Discovered {len(sample_images)} sample produce images.")

    if args.loop:
        print(f"Starting continuous streaming loop (every {args.interval}s)... Press Ctrl+C to stop.\n")
        idx = 0
        try:
            while True:
                img = sample_images[idx % len(sample_images)]
                send_frame(img, args.endpoint, args.device)
                idx += 1
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n🛑 Simulator stopped.")
    else:
        chosen = sample_images[0]
        for s in sample_images:
            if "tomato" in s.name.lower():
                chosen = s
                break
        send_frame(chosen, args.endpoint, args.device)


if __name__ == "__main__":
    main()
