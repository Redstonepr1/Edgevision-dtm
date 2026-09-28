import os
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from edge_vision.config import DATABASE_URL, DATA_DIR
from edge_vision.database.models import Base, ProduceItem, ItemStatus

# SQLite connect args for concurrent reads/writes
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initializes database tables"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    seed_initial_data_if_empty()


def get_db() -> Generator[Session, None, None]:
    """Dependency injection helper for FastAPI routes"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_decay_curve(start_freshness: float, current_freshness: float, days_span: int = 5):
    """Generates synthetic decay curve readings for visualization"""
    points = []
    now = datetime.now(timezone.utc)
    start_f = float(start_freshness)
    curr_f = float(current_freshness)
    step = (start_f - curr_f) / max(1, days_span * 4)
    
    for i in range(days_span * 4, -1, -1):
        t = now - timedelta(hours=i * 6)
        f = max(5.0, min(100.0, curr_f + (i * step)))
        points.append({
            "timestamp": t.isoformat(),
            "freshness": float(round(f, 1))
        })
    return points


def seed_initial_data_if_empty():
    """Seeds rich sample items if the database is currently empty"""
    db = SessionLocal()
    try:
        count = db.query(ProduceItem).count()
        if count > 0:
            return

        now = datetime.now(timezone.utc)

        seed_items = [
            ProduceItem(
                item_id="item-tom-001",
                category="Tomato",
                confidence_score=94.8,
                freshness_percentage=22.5,
                estimated_days_left=2,
                ocr_expiry_date=None,
                image_url="/static/uploads/fresh_tomato.jpg",
                status=ItemStatus.CRITICAL,
                bounding_box_json=json.dumps([120, 80, 360, 320]),
                device_id="ESP32-CAM-01",
                temperature_c=22.4,
                humidity_rh=68.2,
                decay_history_json=json.dumps(generate_decay_curve(95.0, 22.5, 6)),
                first_detected_at=now - timedelta(days=6),
                last_updated_at=now - timedelta(minutes=10),
            ),
            ProduceItem(
                item_id="item-ban-002",
                category="Banana",
                confidence_score=96.2,
                freshness_percentage=28.0,
                estimated_days_left=1,
                ocr_expiry_date=None,
                image_url="/static/uploads/rotten_banana.jpg",
                status=ItemStatus.CRITICAL,
                bounding_box_json=json.dumps([40, 150, 280, 410]),
                device_id="ESP32-CAM-01",
                temperature_c=23.1,
                humidity_rh=65.0,
                decay_history_json=json.dumps(generate_decay_curve(98.0, 28.0, 5)),
                first_detected_at=now - timedelta(days=5),
                last_updated_at=now - timedelta(minutes=15),
            ),
            ProduceItem(
                item_id="item-let-003",
                category="Lettuce",
                confidence_score=88.4,
                freshness_percentage=48.0,
                estimated_days_left=3,
                ocr_expiry_date=None,
                image_url="/static/uploads/fresh_apple.jpg",
                status=ItemStatus.APPROACHING_SPOILAGE,
                bounding_box_json=json.dumps([220, 90, 480, 350]),
                device_id="ESP32-CAM-01",
                temperature_c=21.0,
                humidity_rh=75.5,
                decay_history_json=json.dumps(generate_decay_curve(92.0, 48.0, 4)),
                first_detected_at=now - timedelta(days=4),
                last_updated_at=now - timedelta(hours=1),
            ),
            ProduceItem(
                item_id="item-app-004",
                category="Apple",
                confidence_score=97.5,
                freshness_percentage=91.5,
                estimated_days_left=14,
                ocr_expiry_date=None,
                image_url="/static/uploads/fresh_apple.jpg",
                status=ItemStatus.FRESH,
                bounding_box_json=json.dumps([160, 200, 390, 440]),
                device_id="ESP32-CAM-01",
                temperature_c=18.5,
                humidity_rh=82.0,
                decay_history_json=json.dumps(generate_decay_curve(98.0, 91.5, 2)),
                first_detected_at=now - timedelta(days=2),
                last_updated_at=now - timedelta(hours=2),
            ),
            ProduceItem(
                item_id="item-pkg-005",
                category="Organic Yogurt",
                confidence_score=93.0,
                freshness_percentage=65.0,
                estimated_days_left=4,
                ocr_expiry_date="2026-09-24",
                image_url="/static/uploads/fresh_mango.jpg",
                status=ItemStatus.APPROACHING_SPOILAGE,
                bounding_box_json=json.dumps([350, 120, 560, 380]),
                device_id="ESP32-CAM-01",
                temperature_c=4.2,
                humidity_rh=85.0,
                decay_history_json=json.dumps(generate_decay_curve(100.0, 65.0, 7)),
                first_detected_at=now - timedelta(days=7),
                last_updated_at=now - timedelta(hours=3),
            ),
        ]

        db.add_all(seed_items)
        db.commit()
        print("[Database] Seeded 5 initial produce and packaged items.")
    except Exception as e:
        db.rollback()
        print(f"[Database Error during seeding] {e}")
    finally:
        db.close()
