"""
Dataset Downloader & Labeler for Edge-Vision
============================================
Downloads images from the open GitHub repository:
zijianchen98/Fruit-freshness-detection-dataset (which contains fresh, normal, and rotten
apples, bananas, and oranges with ground-truth XML annotations).

Then converts them into:
1. sample_images/ (or data/freshness_dataset/)
2. data/training_labels.csv for DecayRegressorNN training
"""
import os
import sys
import csv
import xml.etree.ElementTree as ET
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_DIR = PROJECT_ROOT / "data" / "downloaded_dataset"
TARGET_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = PROJECT_ROOT / "data" / "training_labels.csv"

BASE_RAW = "https://raw.githubusercontent.com/zijianchen98/Fruit-freshness-detection-dataset/main"

# Freshness calibration mapping based on ground-truth label classes
# Classes in dataset: fresh_apple, normal_apple, rotten_apple, fresh_banana, normal_banana, rotten_banana, fresh_orange, normal_orange, rotten_orange
CLASS_CONFIG = {
    "fresh_apple": {"category": "Apple", "freshness": 95.0, "days": 16},
    "normal_apple": {"category": "Apple", "freshness": 65.0, "days": 8},
    "rotten_apple": {"category": "Apple", "freshness": 12.0, "days": 1},
    "fresh_banana": {"category": "Banana", "freshness": 92.0, "days": 6},
    "normal_banana": {"category": "Banana", "freshness": 60.0, "days": 3},
    "rotten_banana": {"category": "Banana", "freshness": 10.0, "days": 1},
    "fresh_orange": {"category": "Orange", "freshness": 94.0, "days": 14},
    "normal_orange": {"category": "Orange", "freshness": 62.0, "days": 6},
    "rotten_orange": {"category": "Orange", "freshness": 15.0, "days": 1},
}

def fetch_file(idx):
    img_name = f"{idx}.jpg"
    xml_name = f"{idx}.xml"
    img_url = f"{BASE_RAW}/JPEGImages/{img_name}"
    xml_url = f"{BASE_RAW}/Annotations/{xml_name}"
    
    img_dest = TARGET_DIR / img_name
    xml_dest = TARGET_DIR / xml_name
    
    try:
        if not xml_dest.exists():
            req = urllib.request.Request(xml_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_dest.write_bytes(resp.read())
                
        # Parse XML to see if it has objects we recognize
        tree = ET.parse(xml_dest)
        root = tree.getroot()
        obj = root.find("object")
        if obj is None:
            return None
        name = obj.find("name")
        if name is None or name.text not in CLASS_CONFIG:
            return None
            
        label = name.text
        
        # Now fetch image
        if not img_dest.exists():
            req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                img_dest.write_bytes(resp.read())
                
        cfg = CLASS_CONFIG[label]
        # Return CSV row entry
        rel_path = f"data/downloaded_dataset/{img_name}"
        return {
            "image_path": rel_path,
            "category": cfg["category"],
            "freshness": cfg["freshness"],
            "days_left": cfg["days"]
        }
    except Exception as e:
        return None

def main(total_images=120):
    print(f"[*] Downloading and cataloging {total_images} produce images with ground truth annotations...")
    results = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_file, i): i for i in range(1, total_images + 1)}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
                if len(results) % 15 == 0:
                    print(f"    -> Collected {len(results)} valid labelled images...")
                    
    print(f"[+] Download complete! Successfully retrieved {len(results)} images.")
    
    # Read existing CSV items to preserve original samples
    existing_rows = []
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r["image_path"].startswith("data/downloaded_dataset/"):
                    existing_rows.append(r)
                    
    combined = existing_rows + results
    print(f"[+] Total samples in dataset: {len(combined)}")
    
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "category", "freshness", "days_left"])
        writer.writeheader()
        writer.writerows(combined)
        
    print(f"[+] Updated training catalog at {CSV_PATH}")

if __name__ == "__main__":
    count = 150
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    main(count)
