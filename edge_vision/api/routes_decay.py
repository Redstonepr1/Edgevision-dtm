from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from edge_vision.database.db_session import get_db
from edge_vision.database.models import ProduceItem

router = APIRouter(prefix="/api/v1/decay", tags=["Decay Trajectories"])


@router.get("/trajectories")
def get_decay_trajectories(db: Session = Depends(get_db)):
    """
    Returns visual decay curves and time-lapse progress metrics for all tracked produce.
    Includes historical readings and forward projected decay trajectories.
    """
    items = db.query(ProduceItem).order_by(ProduceItem.freshness_percentage.asc()).all()
    trajectories = []

    now = datetime.now(timezone.utc)

    for item in items:
        hist = item.decay_history
        current_f = item.freshness_percentage
        days_left = item.estimated_days_left

        # Generate forward projection curve until decay
        projected = []
        step_hours = 12
        total_steps = max(2, int(days_left * 2))
        f_drop_per_step = current_f / float(total_steps)

        for s in range(1, total_steps + 1):
            future_t = now + timedelta(hours=s * step_hours)
            projected_f = max(0.0, current_f - (s * f_drop_per_step))
            projected.append({
                "timestamp": future_t.isoformat(),
                "projected_freshness": round(projected_f, 1)
            })

        # Calculate shelf-life consumption % for time-lapse progress bar
        shelf_life_used_pct = round(100.0 - current_f, 1)

        trajectories.append({
            "item_id": item.item_id,
            "category": item.category,
            "current_freshness": current_f,
            "estimated_days_left": days_left,
            "status": item.status.value if hasattr(item.status, "value") else str(item.status),
            "shelf_life_used_pct": shelf_life_used_pct,
            "ocr_expiry_date": item.ocr_expiry_date,
            "history": hist,
            "projected": projected
        })

    return {
        "count": len(trajectories),
        "trajectories": trajectories
    }
