import enum
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    Enum as SQLEnum,
    Text
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ItemStatus(str, enum.Enum):
    FRESH = "FRESH"
    APPROACHING_SPOILAGE = "APPROACHING_SPOILAGE"
    CRITICAL = "CRITICAL"


class ProduceItem(Base):
    """
    SQLAlchemy ORM Model representing an inventory item on the shelf/pantry.
    """
    __tablename__ = "produce_items"

    item_id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    category = Column(String(64), nullable=False, index=True)
    confidence_score = Column(Float, nullable=False, default=95.0)
    freshness_percentage = Column(Float, nullable=False, default=100.0)
    estimated_days_left = Column(Integer, nullable=False, default=7)
    ocr_expiry_date = Column(String(64), nullable=True)
    image_url = Column(String(512), nullable=False)
    status = Column(SQLEnum(ItemStatus), nullable=False, default=ItemStatus.FRESH)

    # Spatial and Device Metadata
    bounding_box_json = Column(Text, nullable=True, default="[]")
    device_id = Column(String(64), nullable=False, default="ESP32-CAM-01")
    temperature_c = Column(Float, nullable=True)
    humidity_rh = Column(Float, nullable=True)

    # Trajectory & Timestamps
    decay_history_json = Column(Text, nullable=False, default="[]")
    first_detected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    @property
    def bounding_box(self) -> List[float]:
        try:
            return json.loads(self.bounding_box_json or "[]")
        except Exception:
            return []

    @bounding_box.setter
    def bounding_box(self, val: List[float]):
        self.bounding_box_json = json.dumps(val or [])

    @property
    def decay_history(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.decay_history_json or "[]")
        except Exception:
            return []

    @decay_history.setter
    def decay_history(self, val: List[Dict[str, Any]]):
        self.decay_history_json = json.dumps(val or [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "confidence_score": round(self.confidence_score, 1),
            "freshness_percentage": round(self.freshness_percentage, 1),
            "estimated_days_left": int(self.estimated_days_left),
            "ocr_expiry_date": self.ocr_expiry_date,
            "image_url": self.image_url,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "bounding_box": self.bounding_box,
            "decay_history": self.decay_history,
            "device_id": self.device_id,
            "temperature_c": self.temperature_c,
            "humidity_rh": self.humidity_rh,
            "first_detected_at": self.first_detected_at.isoformat() if self.first_detected_at else None,
            "last_updated_at": self.last_updated_at.isoformat() if self.last_updated_at else None,
        }


# ==============================================================================
# Pydantic Schemas for API Serialization & Validation
# ==============================================================================

class ProduceItemSchema(BaseModel):
    item_id: str
    category: str
    confidence_score: float
    freshness_percentage: float
    estimated_days_left: int
    ocr_expiry_date: Optional[str] = None
    image_url: str
    status: ItemStatus
    bounding_box: List[float] = Field(default_factory=list)
    decay_history: List[Dict[str, Any]] = Field(default_factory=list)
    device_id: Optional[str] = "ESP32-CAM-01"
    temperature_c: Optional[float] = None
    humidity_rh: Optional[float] = None
    first_detected_at: Optional[str] = None
    last_updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class RecipeIngredient(BaseModel):
    name: str
    amount: str


class RecipeItem(BaseModel):
    recipe_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    description: str
    prep_time_minutes: int
    cook_time_minutes: int
    expiring_ingredients_used: List[str]
    pantry_ingredients_needed: List[str]
    instructions: List[str]
    waste_prevented_kg: float = 0.5


class RecipeResponse(BaseModel):
    triggered: bool
    trigger_reason: str
    expiring_items: List[Dict[str, Any]]
    recipes: List[RecipeItem]


class IngestionResponse(BaseModel):
    success: bool
    frame_id: str
    timestamp: str
    detected_count: int
    items: List[ProduceItemSchema]
    critical_items_count: int
    recipes_triggered: bool
    annotated_image_url: str


class AnalyticsSummary(BaseModel):
    food_waste_reduction_percent: float = -30.0
    monthly_grocery_savings_usd: float = 68.50
    monthly_grocery_savings_inr: float = 4800.0
    co2_reduction_kg: float = 19.4
    active_items_count: int
    critical_items_count: int
    approaching_spoilage_count: int
    fresh_items_count: int
    total_rescued_items: int = 42
