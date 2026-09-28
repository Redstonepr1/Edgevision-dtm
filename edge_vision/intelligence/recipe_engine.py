import os
import json
import uuid
from typing import List, Dict, Any, Optional
from pydantic import ValidationError
from edge_vision.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    CRITICAL_FRESHNESS_PERCENT,
    CRITICAL_DAYS_LEFT
)
from edge_vision.database.models import RecipeItem, RecipeResponse

# Curated high-impact zero-waste recipes database for resilient fallback
CHEF_RECIPE_LIBRARY = {
    "tomato": [
        RecipeItem(
            recipe_id="rec-tom-01",
            title="Rustic Tomato Basil Shakshuka",
            description="Simmer softening, overripe tomatoes with aromatic garlic, cumin, and poached eggs. Overripe tomatoes break down into an ultra-rich, naturally sweet sauce.",
            prep_time_minutes=8,
            cook_time_minutes=15,
            expiring_ingredients_used=["Ripe Softening Tomatoes"],
            pantry_ingredients_needed=["Eggs", "Olive Oil", "Garlic", "Cumin", "Paprika", "Feta or Toast"],
            instructions=[
                "Coarsely chop softened tomatoes; do not discard the flavorful natural juices.",
                "Heat olive oil in a wide skillet, sauté minced garlic, cumin, and smoked paprika until fragrant.",
                "Add chopped tomatoes and simmer for 8 minutes until a thick, jammy sauce forms.",
                "Make small wells in the sauce and crack eggs directly in.",
                "Cover and cook gently on low heat for 5 minutes until egg whites set but yolks remain soft.",
                "Garnish with basil or chili flakes and serve with crusty toast."
            ],
            waste_prevented_kg=0.45
        ),
        RecipeItem(
            recipe_id="rec-tom-02",
            title="Speedy South-Indian Spiced Tomato Chutney",
            description="A bold, tangy condiment made by blistering decaying or bruised tomatoes with mustard seeds, curry leaves, and dried red chilies.",
            prep_time_minutes=5,
            cook_time_minutes=12,
            expiring_ingredients_used=["Expiring Tomatoes"],
            pantry_ingredients_needed=["Mustard Seeds", "Curry Leaves", "Dried Chilies", "Oil", "Salt", "Turmeric"],
            instructions=[
                "Roughly dice tomatoes and sauté in a saucepan with turmeric and salt until completely soft.",
                "In a small tempering pan, heat vegetable oil and add mustard seeds until they crackle.",
                "Add dried red chilies and fresh curry leaves to release aromatic oils.",
                "Pour the hot aromatic oil over the stewed tomato mash and stir vigorously.",
                "Store in a glass jar in the fridge for up to 10 days; perfect with rice, flatbreads, or sandwiches."
            ],
            waste_prevented_kg=0.40
        ),
        RecipeItem(
            recipe_id="rec-tom-03",
            title="Charred Sheet-Pan Roasted Tomato Soup",
            description="Roasting softening tomatoes concentrates sugars and evaporates excess moisture, producing a velvety soup with deep umami.",
            prep_time_minutes=10,
            cook_time_minutes=25,
            expiring_ingredients_used=["Overripe Tomatoes"],
            pantry_ingredients_needed=["Onion", "Garlic cloves", "Olive Oil", "Vegetable Broth", "Cream or Milk", "Oregano"],
            instructions=[
                "Preheat oven to 200°C (400°F). Toss halved tomatoes, quartered onion, and garlic with olive oil and oregano.",
                "Roast on a baking sheet for 20 minutes until blistered and caramelized around the edges.",
                "Transfer all roasted vegetables and pan juices into a blender with warm vegetable broth.",
                "Blend until silky smooth; swirl in a dash of cream or milk.",
                "Serve piping hot with a grilled cheese sandwich."
            ],
            waste_prevented_kg=0.60
        ),
    ],
    "banana": [
        RecipeItem(
            recipe_id="rec-ban-01",
            title="2-Ingredient Flourless Banana Oat Pancakes",
            description="Black-spotted bananas contain peak fructose and natural starches, making fluffy gluten-free pancakes without added sugar.",
            prep_time_minutes=5,
            cook_time_minutes=8,
            expiring_ingredients_used=["Browning Bananas"],
            pantry_ingredients_needed=["Rolled Oats", "Eggs", "Cinnamon", "Pinch of Salt"],
            instructions=[
                "Mash the spotted bananas in a bowl until completely smooth.",
                "Whisk in 2 eggs, 1 cup of rolled oats, and a dash of ground cinnamon.",
                "Heat a non-stick pan over medium heat with a light film of butter or oil.",
                "Ladle batter and cook 2 minutes per side until golden brown bubbles appear.",
                "Serve warm with a drizzle of honey or peanut butter."
            ],
            waste_prevented_kg=0.35
        ),
        RecipeItem(
            recipe_id="rec-ban-02",
            title="One-Bowl Cinnamon Walnut Banana Bread",
            description="The timeless zero-waste champion: turn dark, heavily specked bananas into an ultra-moist bakery loaf.",
            prep_time_minutes=10,
            cook_time_minutes=45,
            expiring_ingredients_used=["Overripe Bananas (2-3)"],
            pantry_ingredients_needed=["Flour", "Sugar or Honey", "Baking Soda", "Melted Butter", "Walnuts"],
            instructions=[
                "Mash overripe bananas in a large mixing bowl.",
                "Stir in melted butter, sugar, beaten egg, and vanilla extract.",
                "Gently fold in flour, baking soda, and crushed walnuts until just combined.",
                "Pour into a lined loaf tin and bake at 175°C (350°F) for 45 minutes.",
                "Cool for 10 minutes before slicing."
            ],
            waste_prevented_kg=0.50
        ),
        RecipeItem(
            recipe_id="rec-ban-03",
            title="Instant Dairy-Free Banana 'Nice Cream'",
            description="Slice speckled bananas and freeze for 2 hours, then blitz into a rich soft-serve ice cream texture.",
            prep_time_minutes=5,
            cook_time_minutes=2,
            expiring_ingredients_used=["Frozen Overripe Bananas"],
            pantry_ingredients_needed=["Cocoa Powder or Peanut Butter", "Milk or Plant Milk", "Dark Chocolate chips"],
            instructions=[
                "Peel and slice bananas into coins, freeze in a container.",
                "Place frozen coins into a food processor with 2 tbsp of cocoa powder and 2 tbsp milk.",
                "Pulse until creamy and soft-serve consistency forms.",
                "Enjoy immediately or refreeze for scoopable ice cream."
            ],
            waste_prevented_kg=0.30
        ),
    ],
    "lettuce": [
        RecipeItem(
            recipe_id="rec-let-01",
            title="Garlic & Sesame Wilted Lettuce Stir-Fry",
            description="Softening lettuce leaves wilt down into a tender, sweet stir-fry in under 3 minutes with soy and toasted sesame oil.",
            prep_time_minutes=3,
            cook_time_minutes=3,
            expiring_ingredients_used=["Wilting Lettuce Leaves"],
            pantry_ingredients_needed=["Garlic", "Soy Sauce", "Sesame Oil", "Cooking Oil"],
            instructions=[
                "Rinse and dry softening lettuce leaves.",
                "Heat wok or skillet over high heat with 1 tbsp oil and sliced garlic.",
                "Toss in lettuce and flash-fry for 60-90 seconds until glossy and slightly wilted.",
                "Splash with soy sauce and toasted sesame oil; remove from heat immediately.",
                "Serve as a vibrant side dish."
            ],
            waste_prevented_kg=0.30
        ),
        RecipeItem(
            recipe_id="rec-let-02",
            title="Zero-Waste Green Goddess Lettuce Pesto",
            description="Blend expiring salad greens with toasted nuts, garlic, and parmesan for a versatile pasta sauce.",
            prep_time_minutes=5,
            cook_time_minutes=0,
            expiring_ingredients_used=["Expiring Salad Greens & Lettuce"],
            pantry_ingredients_needed=["Walnuts or Almonds", "Garlic", "Olive Oil", "Lemon Juice", "Parmesan or Nutritional Yeast"],
            instructions=[
                "Pack lettuce, nuts, garlic, and lemon juice into a blender.",
                "Stream in olive oil while pulsing until an emulsion forms.",
                "Stir in grated cheese and season with sea salt.",
                "Toss with warm pasta or spread on warm sourdough."
            ],
            waste_prevented_kg=0.25
        ),
        RecipeItem(
            recipe_id="rec-let-03",
            title="Silky French Lettuce & Pea Soup",
            description="Simmer wilted lettuce with sweet green peas and broth for an elegant, vibrant green soup.",
            prep_time_minutes=5,
            cook_time_minutes=15,
            expiring_ingredients_used=["Wilted Lettuce Head"],
            pantry_ingredients_needed=["Frozen Peas", "Vegetable Broth", "Butter or Oil", "Onion", "Cream"],
            instructions=[
                "Sauté diced onion in butter until translucent.",
                "Add shredded lettuce, frozen peas, and vegetable broth; bring to a boil.",
                "Simmer for 8 minutes until greens are tender.",
                "Puree with an immersion blender until emerald green and velvety.",
                "Finish with a spoonful of cream and cracked black pepper."
            ],
            waste_prevented_kg=0.40
        ),
    ]
}


def should_trigger_recipe_engine(item: Dict[str, Any]) -> bool:
    """Returns True if produce item is below 30% freshness or <= 2 days left"""
    freshness = float(item.get("freshness_percentage", 100.0))
    days = int(item.get("estimated_days_left", 10))
    status = str(item.get("status", "")).upper()
    return freshness < CRITICAL_FRESHNESS_PERCENT or days <= CRITICAL_DAYS_LEFT or "CRITICAL" in status


def generate_zero_waste_recipes(expiring_items: List[Dict[str, Any]]) -> RecipeResponse:
    """
    Main LLM Recipe Engine Service.
    Triggers when items fall below 30% freshness or <= 2 days remaining.
    Generates a structured JSON list of 3 tailored recipes.
    """
    # Filter items that meet critical criteria
    critical = [item for item in expiring_items if should_trigger_recipe_engine(item)]

    if not critical:
        return RecipeResponse(
            triggered=False,
            trigger_reason="All monitored items are within acceptable freshness limits (>30% freshness, >2 days).",
            expiring_items=[],
            recipes=[]
        )

    categories = list(set([item.get("category", "Produce").strip() for item in critical]))
    trigger_summary = ", ".join([f"{item.get('category')} ({item.get('freshness_percentage')}%, {item.get('estimated_days_left')}d left)" for item in critical])

    # Attempt LLM generation if GEMINI_API_KEY is available
    if GEMINI_API_KEY:
        try:
            recipes = _call_gemini_recipe_engine(critical, categories)
            if recipes and len(recipes) >= 3:
                return RecipeResponse(
                    triggered=True,
                    trigger_reason=f"Proactive Alert: {len(critical)} item(s) critical [{trigger_summary}]",
                    expiring_items=critical,
                    recipes=recipes[:3]
                )
        except Exception as e:
            print(f"[RecipeEngine Gemini Warning] Fallback triggered due to: {e}")

    # Fallback to authentic zero-waste chef library
    fallback_recipes = _generate_fallback_recipes(categories)
    return RecipeResponse(
        triggered=True,
        trigger_reason=f"Proactive Alert: {len(critical)} item(s) critical [{trigger_summary}]",
        expiring_items=critical,
        recipes=fallback_recipes
    )


def _call_gemini_recipe_engine(critical_items: List[Dict[str, Any]], categories: List[str]) -> Optional[List[RecipeItem]]:
    """Calls Google Gemini API via official google-genai SDK with structured output"""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        item_names = ", ".join(categories)

        prompt = f"""
You are an expert zero-waste chef and food sustainability AI.
The user has the following expiring produce inventory in their refrigerator that is approaching decay (under 30% freshness or <= 2 days left):
{json.dumps(critical_items, indent=2)}

Generate exactly 3 tailored, delicious zero-waste recipes specifically designed to rescue and cook these expiring ingredients ({item_names}) immediately.

Return a JSON array of 3 objects with these exact keys:
[
  {{
    "recipe_id": "rec-01",
    "title": "Recipe Title",
    "description": "Why this rescues overripe/expiring ingredients...",
    "prep_time_minutes": 10,
    "cook_time_minutes": 15,
    "expiring_ingredients_used": ["Tomato", ...],
    "pantry_ingredients_needed": ["Olive oil", "Garlic", ...],
    "instructions": ["Step 1...", "Step 2..."],
    "waste_prevented_kg": 0.45
  }}
]
Ensure the response is ONLY valid raw JSON with no surrounding markdown backticks.
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                response_mime_type="application/json"
            )
        )

        raw_text = response.text.strip()
        data = json.loads(raw_text)
        if isinstance(data, dict) and "recipes" in data:
            data = data["recipes"]

        validated_recipes = []
        for r in data:
            validated_recipes.append(RecipeItem(**r))

        return validated_recipes

    except Exception as e:
        print(f"[Gemini API Error] {e}")
        return None


def _generate_fallback_recipes(categories: List[str]) -> List[RecipeItem]:
    """Generates 3 tailored zero-waste recipes using internal culinary knowledge"""
    recipes = []
    
    # Check for specific category matches
    for cat in categories:
        clean = cat.lower()
        for key in CHEF_RECIPE_LIBRARY:
            if key in clean:
                recipes.extend(CHEF_RECIPE_LIBRARY[key])
                break

    # If we don't have 3, supplement from other categories
    if len(recipes) < 3:
        for cat_list in CHEF_RECIPE_LIBRARY.values():
            for rec in cat_list:
                if rec not in recipes:
                    recipes.append(rec)
                if len(recipes) >= 3:
                    break
            if len(recipes) >= 3:
                break

    return recipes[:3]
