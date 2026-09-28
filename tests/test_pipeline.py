import os
import unittest
import numpy as np
import cv2
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient

from edge_vision.main import app
from edge_vision.database.models import ProduceItem, ItemStatus, RecipeItem
from edge_vision.database.db_session import SessionLocal, init_db
from edge_vision.cv.yolo_detector import YOLOProduceDetector
from edge_vision.cv.ocr_engine import ExpiryOCREngine
from edge_vision.cv.decay_model import PyTorchDecayModel
from edge_vision.intelligence.recipe_engine import (
    generate_zero_waste_recipes,
    should_trigger_recipe_engine
)


class TestEdgeVisionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)
        cls.yolo = YOLOProduceDetector()
        cls.ocr = ExpiryOCREngine()
        cls.decay = PyTorchDecayModel()

    def test_01_health_endpoint(self):
        """Test server health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["platform"], "Edge-Vision")

    def test_02_database_crud(self):
        """Test database models and status persistence"""
        db = SessionLocal()
        try:
            test_item = ProduceItem(
                item_id="test-item-123",
                category="Tomato",
                confidence_score=94.5,
                freshness_percentage=25.0,
                estimated_days_left=2,
                image_url="/static/uploads/test.jpg",
                status=ItemStatus.CRITICAL
            )
            db.merge(test_item)
            db.commit()

            fetched = db.query(ProduceItem).filter(ProduceItem.item_id == "test-item-123").first()
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched.category, "Tomato")
            self.assertEqual(fetched.status, ItemStatus.CRITICAL)
            self.assertEqual(fetched.estimated_days_left, 2)

            # Cleanup
            db.delete(fetched)
            db.commit()
        finally:
            db.close()

    def test_03_ocr_date_calculation(self):
        """Test OCR date parsing and days left calculations"""
        future_date = (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%d/%m/%Y")
        days = self.ocr._calculate_days_left(future_date)
        self.assertIsNotNone(days)
        self.assertIn(days, [4, 5, 6])

        # Test month format
        month_date = "25-OCT-2026"
        days_oct = self.ocr._calculate_days_left(month_date)
        self.assertIsNotNone(days_oct)

    def test_04_pytorch_decay_model(self):
        """Test PyTorch decay regression model feature extraction and inference"""
        # Create synthetic produce crop (200x200 BGR)
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        crop[:, :] = (30, 40, 210)  # Red tomato color

        feats = self.decay.extract_decay_features(crop, "Tomato")
        self.assertEqual(len(feats), 16)

        pred = self.decay.predict_decay(crop, "Tomato", temperature_c=23.0)
        self.assertIn("freshness_percentage", pred)
        self.assertIn("estimated_days_left", pred)
        self.assertIn("status", pred)
        self.assertIsInstance(pred["status"], ItemStatus)
        self.assertGreater(pred["freshness_percentage"], 0.0)

    def test_05_yolo_produce_detector(self):
        """Test YOLOv8 produce detector on sample image or synthetic frame"""
        test_img_path = "sample_images/fresh_tomato.jpg"
        if os.path.exists(test_img_path):
            frame = cv2.imread(test_img_path)
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[100:380, 150:450] = (20, 20, 220)

        detections = self.yolo.detect_produce(frame)
        self.assertIsInstance(detections, list)
        self.assertGreater(len(detections), 0)
        first_det = detections[0]
        self.assertIn("box", first_det)
        self.assertIn("category", first_det)
        self.assertIn("confidence", first_det)

        # Test frame annotation HUD
        annotated = self.yolo.annotate_frame(frame, detections)
        self.assertEqual(annotated.shape, frame.shape)

    def test_06_recipe_engine(self):
        """Test LLM Recipe Engine trigger logic and structured JSON recipes"""
        # Non-critical item: freshness 85%, 10 days left
        fresh_item = {"category": "Tomato", "freshness_percentage": 85.0, "estimated_days_left": 10}
        self.assertFalse(should_trigger_recipe_engine(fresh_item))

        # Critical item: freshness 22%, 2 days left
        critical_item = {"category": "Tomato", "freshness_percentage": 22.0, "estimated_days_left": 2, "status": "CRITICAL"}
        self.assertTrue(should_trigger_recipe_engine(critical_item))

        # Generate recipes
        resp = generate_zero_waste_recipes([critical_item])
        self.assertTrue(resp.triggered)
        self.assertEqual(len(resp.recipes), 3)

        for rec in resp.recipes:
            self.assertIsInstance(rec, RecipeItem)
            self.assertTrue(len(rec.title) > 0)
            self.assertTrue(len(rec.instructions) > 0)
            self.assertGreater(rec.waste_prevented_kg, 0.0)

    def test_07_api_endpoints(self):
        """Test inventory, analytics, recipes, and decay endpoints"""
        # Inventory GET
        r_inv = self.client.get("/api/v1/inventory")
        self.assertEqual(r_inv.status_code, 200)
        items = r_inv.json()
        self.assertIsInstance(items, list)

        # Analytics GET
        r_ana = self.client.get("/api/v1/analytics")
        self.assertEqual(r_ana.status_code, 200)
        ana_data = r_ana.json()
        self.assertEqual(ana_data["food_waste_reduction_percent"], -30.0)
        self.assertIn("monthly_grocery_savings_usd", ana_data)

        # Recipes Expiring GET
        r_rec = self.client.get("/api/v1/recipes/expiring")
        self.assertEqual(r_rec.status_code, 200)
        rec_data = r_rec.json()
        self.assertIn("triggered", rec_data)

        # Decay Trajectories GET
        r_dec = self.client.get("/api/v1/decay/trajectories")
        self.assertEqual(r_dec.status_code, 200)
        dec_data = r_dec.json()
        self.assertIn("trajectories", dec_data)

    def test_08_frame_ingestion_endpoint(self):
        """Test ESP32-CAM frame ingestion endpoint POST /api/v1/ingest/frame"""
        test_img_path = "sample_images/fresh_tomato.jpg"
        if not os.path.exists(test_img_path):
            test_img_path = "sample_images/fresh_apple.jpg"

        with open(test_img_path, "rb") as f:
            response = self.client.post(
                "/api/v1/ingest/frame",
                files={"file": (os.path.basename(test_img_path), f, "image/jpeg")},
                data={"device_id": "TEST-ESP32-NODE", "temperature_c": "23.4", "humidity_rh": "66.0"}
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertGreater(data["detected_count"], 0)
        self.assertIn("items", data)
        self.assertIn("annotated_image_url", data)

    def test_09_deduplication_and_ingestion_tracking(self):
        """Test that successive frame ingestions update existing items without creating duplicates"""
        test_img_path = "sample_images/fresh_tomato.jpg"
        if not os.path.exists(test_img_path):
            test_img_path = "sample_images/fresh_apple.jpg"

        test_device = "TEST-DEDUP-CAM"

        # Frame 1: Ingest first frame
        with open(test_img_path, "rb") as f:
            resp1 = self.client.post(
                "/api/v1/ingest/frame",
                files={"file": (os.path.basename(test_img_path), f, "image/jpeg")},
                data={"device_id": test_device}
            )
        self.assertEqual(resp1.status_code, 200)
        items1 = resp1.json()["items"]
        self.assertGreater(len(items1), 0)
        first_item_id = items1[0]["item_id"]

        # Frame 2: Ingest second frame from same device
        with open(test_img_path, "rb") as f:
            resp2 = self.client.post(
                "/api/v1/ingest/frame",
                files={"file": (os.path.basename(test_img_path), f, "image/jpeg")},
                data={"device_id": test_device}
            )
        self.assertEqual(resp2.status_code, 200)
        items2 = resp2.json()["items"]
        # The same item ID should have been updated rather than a new duplicate inserted
        self.assertEqual(items2[0]["item_id"], first_item_id)

        # Test deduplicate endpoint
        dedup_resp = self.client.post("/api/v1/inventory/deduplicate")
        self.assertEqual(dedup_resp.status_code, 200)
        self.assertTrue(dedup_resp.json()["success"])


if __name__ == "__main__":
    unittest.main()
