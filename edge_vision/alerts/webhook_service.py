import time
import requests
from typing import Dict, Any, List
from edge_vision.config import AMBIENT_WEBHOOK_URL

# In-memory recent notification event log for UI polling
NOTIFICATION_HISTORY: List[Dict[str, Any]] = []


def send_ambient_alert(
    event_type: str,
    title: str,
    message: str,
    items: List[Dict[str, Any]],
    channel: str = "WhatsApp / Webhook"
) -> Dict[str, Any]:
    """
    Dispatches ambient notification to external webhooks (e.g. WhatsApp / ntfy / Slack)
    and registers it in the local notification event stream.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    alert_record = {
        "alert_id": f"alt-{int(time.time() * 1000)}",
        "timestamp": timestamp,
        "event_type": event_type,
        "title": title,
        "message": message,
        "channel": channel,
        "items": items,
        "delivered": False
    }

    # If external webhook URL configured, dispatch HTTP POST
    if AMBIENT_WEBHOOK_URL:
        try:
            resp = requests.post(
                AMBIENT_WEBHOOK_URL,
                json=alert_record,
                headers={"Content-Type": "application/json", "User-Agent": "Edge-Vision-Alert/1.0"},
                timeout=4
            )
            alert_record["delivered"] = resp.status_code in [200, 201, 202, 204]
            alert_record["status_code"] = resp.status_code
        except Exception as e:
            alert_record["delivery_error"] = str(e)
    else:
        # Simulated ambient channel success
        alert_record["delivered"] = True
        alert_record["status_code"] = 200

    NOTIFICATION_HISTORY.insert(0, alert_record)
    # Keep last 50 notifications
    if len(NOTIFICATION_HISTORY) > 50:
        NOTIFICATION_HISTORY.pop()

    print(f"[Ambient Alert] {channel} ➡️ {title}: {message}")
    return alert_record


def get_recent_alerts(limit: int = 10) -> List[Dict[str, Any]]:
    """Returns the most recent ambient alerts"""
    return NOTIFICATION_HISTORY[:limit]
