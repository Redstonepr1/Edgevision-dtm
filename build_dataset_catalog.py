"""
Catalog Generator for Edge-Vision
================================
Parses XML annotations from data/fruit_repo/Annotations and indexes
real produce images (Apples, Bananas, Oranges across Fresh, Normal, Rotten stages).
Creates data/training_labels.csv with hundreds of labelled samples.
"""
import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_DIR = PROJECT_ROOT / "data" / "fruit_repo"
ANNOTATIONS_DIR = REPO_DIR / "Annotations"
IMAGES_DIR = REPO_DIR / "JPEGImages"
CSV_PATH = PROJECT_ROOT / "data" / "training_labels.csv"

# Label to category & ground-truth freshness mapping
CLASS_MAP = {
    "fresh_apple":    {"category": "Apple",  "freshness": 95.0, "days_left": 16},
    "normal_apple":   {"category": "Apple",  "freshness": 68.0, "days_left": 8},
    "rotten_apple":   {"category": "Apple",  "freshness": 12.0, "days_left": 1},
    "fresh_banana":   {"category": "Banana", "freshness": 94.0, "days_left": 6},
    "normal_banana":  {"category": "Banana", "freshness": 62.0, "days_left": 3},
    "rotten_banana":  {"category": "Banana", "freshness": 10.0, "days_left": 1},
    "fresh_orange":   {"category": "Orange", "freshness": 93.0, "days_left": 14},
    "normal_orange":  {"category": "Orange", "freshness": 65.0, "days_left": 7},
    "rotten_orange":  {"category": "Orange", "freshness": 14.0, "days_left": 1},
}

def generate_catalog(max_samples_per_class=60):
    class_counts = {k: 0 for k in CLASS_MAP}
    catalog_rows = []

    # Preserve initial sample images (tomatoes, strawberries, mangoes etc.)
    original_samples = [
        {"image_path": "sample_images/fresh_tomato.jpg", "category": "Tomato", "freshness": 94.0, "days_left": 8},
        {"image_path": "sample_images/rotten_tomato.jpg", "category": "Tomato", "freshness": 7.0, "days_left": 1},
        {"image_path": "sample_images/fresh_mango.jpg", "category": "Produce Item", "freshness": 90.0, "days_left": 7},
        {"image_path": "sample_images/rotten_mango.jpg", "category": "Produce Item", "freshness": 12.0, "days_left": 1},
        {"image_path": "sample_images/fresh_strawberry.jpg", "category": "Produce Item", "freshness": 93.0, "days_left": 4},
        {"image_path": "sample_images/rotten_strawberry.jpg", "category": "Produce Item", "freshness": 9.0, "days_left": 1},
    ]
    catalog_rows.extend(original_samples)

    xml_files = sorted(list(ANNOTATIONS_DIR.glob("*.xml")), key=lambda p: int(p.stem) if p.stem.isdigit() else 999999)

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj = root.find("object")
            if obj is None:
                continue
            name_el = obj.find("name")
            if name_el is None or name_el.text not in CLASS_MAP:
                continue

            label = name_el.text
            if class_counts[label] >= max_samples_per_class:
                continue

            img_name = f"{xml_path.stem}.jpg"
            img_path = IMAGES_DIR / img_name
            if not img_path.exists():
                continue

            mapping = CLASS_MAP[label]
            rel_img_path = f"data/fruit_repo/JPEGImages/{img_name}"

            catalog_rows.append({
                "image_path": rel_img_path,
                "category": mapping["category"],
                "freshness": mapping["freshness"],
                "days_left": mapping["days_left"]
            })
            class_counts[label] += 1
        except Exception:
            continue

    print(f"[*] Distribution across classes:")
    for lbl, count in class_counts.items():
        print(f"    - {lbl:15s}: {count} images")

    print(f"[*] Total indexed images: {len(catalog_rows)}")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "category", "freshness", "days_left"])
        writer.writeheader()
        writer.writerows(catalog_rows)

    print(f"[+] Successfully wrote {len(catalog_rows)} rows to {CSV_PATH}")

if __name__ == "__main__":
    generate_catalog(max_samples_per_class=50)
