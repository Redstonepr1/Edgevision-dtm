from typing import Optional, Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel
from edge_vision.alerts.webhook_service import send_ambient_alert, get_recent_alerts

router = APIRouter(prefix="/api/v1/alerts", tags=["Ambient Alerts"])


class TestAlertRequest(BaseModel):
    title: str = "Produce Spoilage Warning"
    message: str = "2 Tomatoes & 1 Banana have crossed critical freshness (<30%). Immediate cooking recommended."
    channel: str = "WhatsApp Ambient Alert"


@router.post("/webhook")
def trigger_alert(req: TestAlertRequest):
    """Triggers an ambient alert webhook notification"""
    alert = send_ambient_alert(
        event_type="MANUAL_TEST_ALERT",
        title=req.title,
        message=req.message,
        items=[],
        channel=req.channel
    )
    return {"success": True, "alert": alert}


@router.get("")
def list_alerts(limit: int = 10):
    """Lists recent ambient notifications"""
    return {"alerts": get_recent_alerts(limit=limit)}
