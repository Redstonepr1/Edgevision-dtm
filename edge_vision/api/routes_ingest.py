import os
import time
import uuid
import json
import cv2
import numpy as np
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Request, Depends, HTTPException
from sqlalchemy.orm import Session

from edge_vision.config import UPLOAD_DIR, DEFAULT_DEVICE_ID
from edge_vision.database.db_session import get_db, generate_decay_curve
from edge_vision.database.models import (
    ProduceItem,
    ProduceItemSchema,
    ItemStatus,
    IngestionResponse
)
from edge_vision.cv.yolo_detector import YOLOProduceDetector, compute_box_iou
from edge_vision.cv.ocr_engine import ExpiryOCREngine
from edge_vision.cv.decay_model import PyTorchDecayModel
from edge_vision.intelligence.recipe_engine import generate_zero_waste_recipes, should_trigger_recipe_engine
from edge_vision.alerts.webhook_service import send_ambient_alert

router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])

# Persistent CV Singletons
yolo_detector = YOLOProduceDetector()
ocr_engine = ExpiryOCREngine()
decay_model = PyTorchDecayModel()


@router.post("/frame", response_model=IngestionResponse)
async def ingest_frame(
    request: Request,
    file: Optional[UploadFile] = File(None),
    device_id: Optional[str] = Form(DEFAULT_DEVICE_ID),
    temperature_c: Optional[float] = Form(None),
    humidity_rh: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """
    ESP32-CAM Scheduled Image Capture Ingestion Endpoint.
    Accepts image frame via multipart form upload or raw binary request body.
    Processes sequentially: YOLOv8 -> OCR -> PyTorch Decay Regression -> DB -> Recipe Trigger.
    """
    frame_bytes = None

    if file is not None:
        frame_bytes = await file.read()
    else:
        # Check raw request body (common for ESP32 raw HTTP client)
        body = await request.body()
        if body and len(body) > 100:
            frame_bytes = body

    if not frame_bytes:
        raise HTTPException(status_code=400, detail="No image frame payload received.")

    # Decode image to OpenCV BGR numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Failed to decode image frame.")

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    frame_id = f"frame_{timestamp_str}_{str(uuid.uuid4())[:6]}"
    raw_filename = f"{frame_id}_raw.jpg"
    annotated_filename = f"{frame_id}_annotated.jpg"

    raw_path = UPLOAD_DIR / raw_filename
    annotated_path = UPLOAD_DIR / annotated_filename

    # Save raw frame
    cv2.imwrite(str(raw_path), frame)

    # 1. YOLOv8 Object Detection & Classification
    detections = yolo_detector.detect_produce(frame)

    processed_items = []
    critical_items = []
    items_meta_for_hud = []

    now = datetime.now(timezone.utc)
    effective_device_id = device_id or DEFAULT_DEVICE_ID
    existing_items = db.query(ProduceItem).filter(ProduceItem.device_id == effective_device_id).all()

    for idx, det in enumerate(detections):
        box = det["box"]
        category = det["category"]
        conf = det["confidence"]
        crop = det["crop"]

        # 2. OCR Expiry Date Extraction for packaged goods
        ocr_result = {"expiry_date": None, "days_left": None}
        if "Package" in category or "Milk" in category or "Yogurt" in category or "Salad" in category:
            ocr_result = ocr_engine.parse_expiry_date(crop)

        # 3. PyTorch Decay Regression Model
        decay_res = decay_model.predict_decay(
            crop_bgr=crop,
            category=category,
            temperature_c=temperature_c,
            humidity_rh=humidity_rh
        )

        freshness = decay_res["freshness_percentage"]
        est_days = decay_res["estimated_days_left"]
        status = decay_res["status"]

        # If OCR detected an explicit expiry date, adjust days remaining
        ocr_date = ocr_result.get("expiry_date")
        if ocr_result.get("days_left") is not None:
            est_days = ocr_result["days_left"]
            if est_days <= 2:
                status = ItemStatus.CRITICAL
                freshness = min(freshness, 25.0)

        # Check for existing item on this device to update rather than creating duplicates
        matched_item = None
        for cand in existing_items:
            # Match by category and bounding box overlap (IoU > 0.15)
            if cand.category.lower() == category.lower():
                cand_box = cand.bounding_box
                iou_score = compute_box_iou(box, cand_box)
                if iou_score > 0.15:
                    matched_item = cand
                    break
        
        # If no IoU match but only one item of that category exists for this device, treat as same item
        if not matched_item:
            category_candidates = [c for c in existing_items if c.category.lower() == category.lower()]
            if len(category_candidates) == 1:
                matched_item = category_candidates[0]

        crop_filename = f"{frame_id}_crop_{idx}_{category.lower()}.jpg"
        crop_path = UPLOAD_DIR / crop_filename
        if crop is not None and crop.size > 0:
            cv2.imwrite(str(crop_path), crop)
            item_image_url = f"/static/uploads/{crop_filename}"
        else:
            item_image_url = f"/static/uploads/{raw_filename}"

        if matched_item:
            # Update existing item
            matched_item.confidence_score = float(conf)
            matched_item.freshness_percentage = float(freshness)
            matched_item.estimated_days_left = int(est_days)
            if ocr_date:
                matched_item.ocr_expiry_date = ocr_date
            matched_item.image_url = item_image_url
            matched_item.status = status
            matched_item.bounding_box_json = json.dumps([float(v) for v in box])
            if temperature_c is not None:
                matched_item.temperature_c = float(temperature_c)
            if humidity_rh is not None:
                matched_item.humidity_rh = float(humidity_rh)
            matched_item.last_updated_at = now

            # Append current reading to decay history
            try:
                hist = json.loads(matched_item.decay_history_json or "[]")
            except Exception:
                hist = []
            hist.append({"timestamp": now.isoformat(), "freshness": float(freshness)})
            if len(hist) > 10:
                hist = hist[-10:]
            matched_item.decay_history_json = json.dumps(hist)

            db.commit()
            db.refresh(matched_item)
            db_item = matched_item
        else:
            # Insert new item
            item_id = f"item-{category.lower()[:3]}-{str(uuid.uuid4())[:6]}"
            decay_history = generate_decay_curve(
                start_freshness=min(100.0, freshness + 35.0),
                current_freshness=freshness,
                days_span=4
            )

            db_item = ProduceItem(
                item_id=item_id,
                category=category,
                confidence_score=float(conf),
                freshness_percentage=float(freshness),
                estimated_days_left=int(est_days),
                ocr_expiry_date=ocr_date,
                image_url=item_image_url,
                status=status,
                bounding_box_json=json.dumps([float(v) for v in box]),
                device_id=effective_device_id,
                temperature_c=float(temperature_c) if temperature_c is not None else None,
                humidity_rh=float(humidity_rh) if humidity_rh is not None else None,
                decay_history_json=json.dumps(decay_history),
                first_detected_at=now,
                last_updated_at=now
            )
            db.add(db_item)
            db.commit()
            db.refresh(db_item)
            existing_items.append(db_item)

        item_dict = db_item.to_dict()
        processed_items.append(ProduceItemSchema(**item_dict))

        if status == ItemStatus.CRITICAL:
            critical_items.append(item_dict)

        items_meta_for_hud.append({
            "box": box,
            "category": category,
            "confidence_score": conf,
            "freshness_percentage": freshness,
            "status": status.value,
            "estimated_days_left": est_days
        })

    # 4. Check Zero-Waste Recipe Engine Trigger & Ambient Alerts
    recipes_triggered = False
    if critical_items:
        recipes_triggered = True
        recipe_resp = generate_zero_waste_recipes(critical_items)
        # Send Ambient / WhatsApp Alert Trigger
        item_names = ", ".join([f"{i['category']} ({i['estimated_days_left']}d left)" for i in critical_items])
        send_ambient_alert(
            event_type="EXPIRING_PRODUCE_ALERT",
            title="Produce Spoilage Warning",
            message=f"{len(critical_items)} produce item(s) are approaching decay: {item_names}. 3 zero-waste recipes have been generated!",
            items=critical_items,
            channel="WhatsApp Ambient Alert"
        )

    # 5. Generate Annotated Frame with Edge-Vision HUD Badges
    annotated_frame = yolo_detector.annotate_frame(frame, items_meta_for_hud, device_id=device_id or DEFAULT_DEVICE_ID)
    cv2.imwrite(str(annotated_path), annotated_frame)
    annotated_url = f"/static/uploads/{annotated_filename}"

    return IngestionResponse(
        success=True,
        frame_id=frame_id,
        timestamp=now.isoformat(),
        detected_count=len(processed_items),
        items=processed_items,
        critical_items_count=len(critical_items),
        recipes_triggered=recipes_triggered,
        annotated_image_url=annotated_url
    )
