from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from edge_vision.database.db_session import get_db
from edge_vision.database.models import ProduceItem, ItemStatus, RecipeResponse
from edge_vision.intelligence.recipe_engine import generate_zero_waste_recipes, should_trigger_recipe_engine

router = APIRouter(prefix="/api/v1/recipes", tags=["Recipes"])


@router.get("/expiring", response_model=RecipeResponse)
def get_expiring_recipes(db: Session = Depends(get_db)):
    """
    Checks all inventory items and triggers the LLM Recipe Engine
    if any item has <30% freshness or <= 2 days remaining.
    Returns 3 tailored zero-waste recipes.
    """
    items = db.query(ProduceItem).all()
    item_dicts = [i.to_dict() for i in items]
    response = generate_zero_waste_recipes(item_dicts)
    return response


@router.post("/generate-custom", response_model=RecipeResponse)
def generate_custom_recipes(categories: List[str]):
    """Manually triggers recipe generation for specified ingredients"""
    simulated_items = [
        {"category": cat, "freshness_percentage": 20.0, "estimated_days_left": 1, "status": "CRITICAL"}
        for cat in categories
    ]
    return generate_zero_waste_recipes(simulated_items)
