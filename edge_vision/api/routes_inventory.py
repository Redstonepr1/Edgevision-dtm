from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from edge_vision.database.db_session import get_db, seed_initial_data_if_empty
from edge_vision.database.models import ProduceItem, ProduceItemSchema, ItemStatus

router = APIRouter(prefix="/api/v1/inventory", tags=["Inventory"])


@router.get("", response_model=List[ProduceItemSchema])
def get_inventory(
    status: Optional[str] = Query(None, description="Filter by status: FRESH, APPROACHING_SPOILAGE, CRITICAL"),
    db: Session = Depends(get_db)
):
    """Returns all current produce and packaged goods in shelf inventory"""
    query = db.query(ProduceItem)
    if status and status.upper() != "ALL":
        try:
            status_enum = ItemStatus(status.upper())
            query = query.filter(ProduceItem.status == status_enum)
        except ValueError:
            pass

    items = query.order_by(ProduceItem.estimated_days_left.asc()).all()
    return [ProduceItemSchema(**item.to_dict()) for item in items]


@router.get("/{item_id}", response_model=ProduceItemSchema)
def get_item(item_id: str, db: Session = Depends(get_db)):
    """Returns single item details with decay trajectory history"""
    item = db.query(ProduceItem).filter(ProduceItem.item_id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found.")
    return ProduceItemSchema(**item.to_dict())


@router.delete("/{item_id}")
def delete_item(item_id: str, db: Session = Depends(get_db)):
    """Removes an item after cooking/consumption"""
    item = db.query(ProduceItem).filter(ProduceItem.item_id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found.")
    db.delete(item)
    db.commit()
    return {"success": True, "message": f"Item {item_id} marked as consumed/rescued."}


@router.post("/deduplicate")
def deduplicate_inventory(db: Session = Depends(get_db)):
    """
    Cleans up duplicate items in SQLite inventory.
    Groups items by device and category/location, keeping the most recently updated item
    and purging older duplicates.
    """
    from collections import defaultdict
    from edge_vision.cv.yolo_detector import compute_box_iou

    all_items = db.query(ProduceItem).order_by(ProduceItem.last_updated_at.desc()).all()
    if not all_items:
        return {"success": True, "message": "Inventory is empty.", "deleted_count": 0}

    # Group items by device and category
    groups = defaultdict(list)
    for it in all_items:
        groups[(it.device_id, it.category.lower())].append(it)

    deleted_count = 0
    for (dev, cat), items in groups.items():
        if len(items) <= 1:
            continue

        # Cluster items by IoU bounding box proximity
        kept = []
        for it in items:
            box = it.bounding_box
            is_dup = False
            for k in kept:
                k_box = k.bounding_box
                # If both have boxes and overlap > 0.15, or if neither has box / single category overlap
                if box and k_box and compute_box_iou(box, k_box) > 0.15:
                    is_dup = True
                    break
                elif not box and not k_box:
                    is_dup = True
                    break

            if is_dup:
                db.delete(it)
                deleted_count += 1
            else:
                kept.append(it)

    db.commit()
    return {
        "success": True,
        "message": f"Successfully purged {deleted_count} duplicate item(s).",
        "deleted_count": deleted_count
    }


@router.post("/reset-sample")
def reset_sample_inventory(db: Session = Depends(get_db)):
    """Clears and re-seeds sample items for quick demoing"""
    db.query(ProduceItem).delete()
    db.commit()
    seed_initial_data_if_empty()
    return {"success": True, "message": "Sample inventory re-seeded."}
