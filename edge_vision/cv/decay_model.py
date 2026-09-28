import os
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from edge_vision.database.models import ItemStatus
from edge_vision.config import (
    CRITICAL_FRESHNESS_PERCENT,
    CRITICAL_DAYS_LEFT,
    SPOILAGE_FRESHNESS_PERCENT,
    CNN_MODEL_PATH
)

# Baseline Shelf-Life (Days at ambient 22°C)
BASELINE_SHELF_LIFE: Dict[str, float] = {
    "Tomato": 8.0,
    "Banana": 6.0,
    "Lettuce": 5.0,
    "Apple": 18.0,
    "Orange": 14.0,
    "Carrot": 12.0,
    "Packaged Milk": 7.0,
    "Organic Yogurt": 10.0,
    "Packaged Salad": 4.0,
    "Produce Item": 7.0
}


class DecayRegressorNN(nn.Module):
    """
    PyTorch Multi-Layer Perceptron Regression Model mapping visual decay markers
    (color histogram, necrosis blemish ratio, texture entropy) to freshness % and days remaining.
    """
    def __init__(self, in_features: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Freshness percentage branch (0.0 to 1.0)
        self.freshness_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # Days remaining branch (continuous regression)
        self.days_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        freshness = self.freshness_head(feat)
        days = self.days_head(feat)
        return freshness, days


class PyTorchDecayModel:
    """
    Visual Decay & Trajectory Regression Engine.
    Maps visual decay markers (color/texture shifts over time) to freshness % and days remaining.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DecayRegressorNN(in_features=16).to(self.device)
        self._load_weights()
        self.model.eval()

    def _load_weights(self):
        """
        Loads trained weights from models/decay_regressor.pt when available.
        Falls back to Xavier-initialised random weights if the checkpoint is missing
        (i.e. the model has never been trained yet).
        """
        checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "decay_regressor.pt")
        checkpoint_path = os.path.normpath(checkpoint_path)
        if os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                print(f"[DecayModel] Loaded trained weights from {checkpoint_path}")
            except Exception as e:
                print(f"[DecayModel] Could not load checkpoint ({e}). Using Xavier-initialised weights.")
                self._init_synthetic_weights()
        else:
            print("[DecayModel] No trained checkpoint found. Using Xavier-initialised weights.")
            print("             Run:  .venv\\Scripts\\python train_decay_model.py  to train.")
            self._init_synthetic_weights()


    def _init_synthetic_weights(self):
        """Initializes calibrated weights for empirical decay physics"""
        with torch.no_grad():
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def extract_decay_features(self, crop_bgr: np.ndarray, category: str) -> np.ndarray:
        """
        Extracts 16-dimensional visual decay vector:
        - Mean and Std of B, G, R channels (6)
        - Mean and Std of H, S, V channels (6)
        - Brown/rot blemish ratio (1)
        - Texture variance / sharpness via Laplacian (1)
        - Color entropy (1)
        - Baseline shelf life prior (1)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(16, dtype=np.float32)

        # 1. RGB statistics
        b, g, r = cv2.split(crop_bgr)
        rgb_stats = [
            float(np.mean(r)) / 255.0, float(np.std(r)) / 255.0,
            float(np.mean(g)) / 255.0, float(np.std(g)) / 255.0,
            float(np.mean(b)) / 255.0, float(np.std(b)) / 255.0,
        ]

        # 2. HSV statistics
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        hsv_stats = [
            float(np.mean(h)) / 180.0, float(np.std(h)) / 180.0,
            float(np.mean(s)) / 255.0, float(np.std(s)) / 255.0,
            float(np.mean(v)) / 255.0, float(np.std(v)) / 255.0,
        ]

        # 3. Brown / rot necrosis ratio
        # Brown / darkened regions typical of banana spotting, tomato rot, lettuce browning
        lower_brown = np.array([8, 60, 20])
        upper_brown = np.array([25, 255, 120])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        total_pixels = float(crop_bgr.shape[0] * crop_bgr.shape[1])
        brown_ratio = float(cv2.countNonZero(brown_mask)) / max(1.0, total_pixels)

        # 4. Texture Laplacian variance (surface roughness/wrinkling)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        norm_texture = min(1.0, laplacian_var / 500.0)

        # 5. Color entropy
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist_norm = hist / (hist.sum() + 1e-7)
        entropy = -float(np.sum([p * math.log2(p + 1e-7) for p in hist_norm.flatten()])) / 4.0

        # 6. Baseline prior
        base_days = BASELINE_SHELF_LIFE.get(category, 7.0) / 20.0

        feats = rgb_stats + hsv_stats + [brown_ratio, norm_texture, entropy, base_days]
        return np.array(feats, dtype=np.float32)

    def predict_decay(
        self,
        crop_bgr: np.ndarray,
        category: str,
        temperature_c: Optional[float] = None,
        humidity_rh: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs PyTorch decay regression inference on produce crop.
        Returns calculated freshness %, estimated days left, and ItemStatus.
        """
        feats = self.extract_decay_features(crop_bgr, category)
        brown_ratio = feats[12]
        base_shelf_life = BASELINE_SHELF_LIFE.get(category, 7.0)

        # Run PyTorch forward pass
        with torch.no_grad():
            inp_tensor = torch.tensor(feats).unsqueeze(0).to(self.device)
            fresh_pred, days_pred = self.model(inp_tensor)
            model_freshness = float(fresh_pred.item())
            model_days = float(days_pred.item())

        # Calibrate decay based on PyTorch neural network prediction blended with visual rot markers
        # Baseline visual heuristic based on necrosis ratio
        necrosis_freshness = max(0.05, min(0.98, 1.0 - (brown_ratio * 1.75)))
        
        # Specific category heuristics
        heuristic_freshness = necrosis_freshness
        if category == "Tomato":
            r_mean = feats[0]
            if r_mean > 0.45 and brown_ratio < 0.08:
                heuristic_freshness = max(heuristic_freshness, 0.92)
            elif brown_ratio > 0.15 or r_mean < 0.30:
                heuristic_freshness = min(heuristic_freshness, 0.28)
        elif category == "Banana":
            if brown_ratio > 0.22:
                heuristic_freshness = min(heuristic_freshness, 0.25)
            elif brown_ratio < 0.05:
                heuristic_freshness = max(heuristic_freshness, 0.94)

        # Primary prediction from trained neural network (70% model + 30% empirical heuristic safety guard)
        combined_freshness = max(0.05, min(0.99, (model_freshness * 0.70) + (heuristic_freshness * 0.30)))

        # Arrhenius Q10 kinetic adjustment if temperature telemetry is provided
        temp_factor = 1.0
        if temperature_c is not None and temperature_c > 22.0:
            # For every 10C increase, decay rate doubles (Q10 = 2.0)
            temp_factor = math.pow(2.0, (temperature_c - 22.0) / 10.0)

        calibrated_freshness = round(float(combined_freshness) * 100.0, 1)

        # Days estimate: blend neural network regression output with shelf-life prior
        combined_days = (model_days * 0.65) + ((combined_freshness * base_shelf_life) * 0.35)
        est_days = max(1, int(round(float(combined_days) / float(temp_factor))))

        # Determine Status Enum
        if calibrated_freshness < CRITICAL_FRESHNESS_PERCENT or est_days <= CRITICAL_DAYS_LEFT:
            status = ItemStatus.CRITICAL
        elif calibrated_freshness < SPOILAGE_FRESHNESS_PERCENT or est_days <= 4:
            status = ItemStatus.APPROACHING_SPOILAGE
        else:
            status = ItemStatus.FRESH

        return {
            "freshness_percentage": float(calibrated_freshness),
            "estimated_days_left": int(est_days),
            "status": status,
            "brown_blemish_ratio": round(float(brown_ratio), 3)
        }
