import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import numpy as np
import cv2

# Regex patterns for expiry labels and dates
EXPIRY_PATTERNS = [
    # E.g. "BEST BEFORE: 24/09/2026", "EXP 15-10-2026", "BB 12.11.2026"
    re.compile(r'(?:BEST\s*BEFORE|EXP(?:IRY)?|USE\s*BY|BB|BBD)[\s.:\-_]*([0-3]?\d[./\-][0-1]?\d[./\-]20\d{2})', re.IGNORECASE),
    # E.g. "EXP: 25 OCT 2026"
    re.compile(r'(?:BEST\s*BEFORE|EXP(?:IRY)?|USE\s*BY|BB|BBD)[\s.:\-_]*([0-3]?\d\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*[\s\-]*20\d{2})', re.IGNORECASE),
    # Generic Date: "28/09/2026"
    re.compile(r'\b([0-3]?\d[./\-][0-1]?\d[./\-]20\d{2})\b'),
]

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}


class ExpiryOCREngine:
    """
    EasyOCR + Regex Date Extractor for Packaged Shelf Goods.
    """
    def __init__(self):
        self.reader = None
        self._initialized = False

    def _get_reader(self):
        """Lazy initialization of EasyOCR reader to optimize cold starts"""
        if not self._initialized:
            try:
                import easyocr
                self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                self._initialized = True
                print("[EasyOCR] Engine initialized successfully.")
            except Exception as e:
                print(f"[EasyOCR Warning] Could not initialize reader: {e}")
                self.reader = None
                self._initialized = True
        return self.reader

    def parse_expiry_date(self, image_crop: np.ndarray) -> Dict[str, Any]:
        """
        Runs OCR on image crop to detect 'Best Before' / Expiry dates.
        Returns extracted date string, days remaining, and OCR text.
        """
        if image_crop is None or image_crop.size == 0:
            return {"expiry_date": None, "days_left": None, "detected_text": []}

        reader = self._get_reader()
        if reader is None:
            return {"expiry_date": None, "days_left": None, "detected_text": []}

        try:
            # Preprocess crop: grayscale + adaptive thresholding for better text contrast
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if len(image_crop.shape) == 3 else image_crop
            
            # Run OCR
            results = reader.readtext(gray, detail=0)
            full_text = " ".join(results).strip()

            date_str = None
            for pattern in EXPIRY_PATTERNS:
                match = pattern.search(full_text)
                if match:
                    date_str = match.group(1).strip()
                    break

            if date_str:
                days_left = self._calculate_days_left(date_str)
                return {
                    "expiry_date": date_str,
                    "days_left": days_left,
                    "detected_text": results
                }

        except Exception as e:
            print(f"[ExpiryOCREngine Error] {e}")

        return {"expiry_date": None, "days_left": None, "detected_text": []}

    def _calculate_days_left(self, date_str: str) -> Optional[int]:
        """Calculates days remaining from extracted date string"""
        clean = re.sub(r'[\s./\-]+', '-', date_str).strip()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        parsed_dt = None

        # Try common date formats
        for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d"):
            try:
                parsed_dt = datetime.strptime(clean, fmt)
                break
            except ValueError:
                pass

        # Try Month Name e.g. 24-Oct-2026
        if not parsed_dt:
            parts = clean.split("-")
            if len(parts) == 3:
                day, month_str, year = parts[0], parts[1].lower()[:3], parts[2]
                if month_str in MONTH_MAP:
                    try:
                        parsed_dt = datetime(int(year), MONTH_MAP[month_str], int(day))
                    except Exception:
                        pass

        if parsed_dt:
            delta = (parsed_dt - now).days
            return max(0, delta)

        return None
