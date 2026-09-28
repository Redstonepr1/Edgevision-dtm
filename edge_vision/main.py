import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from edge_vision.config import STATIC_DIR, TEMPLATES_DIR
from edge_vision.database.db_session import init_db
from edge_vision.api.routes_ingest import router as ingest_router
from edge_vision.api.routes_inventory import router as inventory_router
from edge_vision.api.routes_recipes import router as recipes_router
from edge_vision.api.routes_analytics import router as analytics_router
from edge_vision.api.routes_decay import router as decay_router
from edge_vision.api.routes_alerts import router as alerts_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables and seeds
    print("[Edge-Vision] Starting up platform services...")
    init_db()
    yield
    print("[Edge-Vision] Shutting down.")


app = FastAPI(
    title="Edge-Vision AI",
    description="Zero-Touch AI Food Freshness & Waste Analytics Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount API Routers
app.include_router(ingest_router)
app.include_router(inventory_router)
app.include_router(recipes_router)
app.include_router(analytics_router)
app.include_router(decay_router)
app.include_router(alerts_router)

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the ambient Edge-Vision Web Dashboard"""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "platform": "Edge-Vision",
        "cv_engine": "YOLOv8 + PyTorch Decay Regressor + EasyOCR",
        "intelligence_engine": "LLM Zero-Waste Recipe Engine"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("edge_vision.main:app", host="0.0.0.0", port=8000, reload=True)
