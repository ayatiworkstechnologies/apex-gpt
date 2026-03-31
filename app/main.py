"""
Construction Material Estimator API — v3
=========================================
Endpoints:
  POST /estimate              — structured JSON (with optional city)
  POST /estimate-from-prompt  — plain English (city auto-detected)
  GET  /cities                — list all supported cities
  GET  /health
  GET  /model/info
  GET  /                      — serve frontend SPA

Run  : uvicorn app.main:app --reload
Docs : http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.schemas import (
    EstimateRequest, EstimateResponse,
    PromptRequest, PromptResponse,
    MaterialQuantities, CostEstimate, CostBreakdown, ParsedDetails,
)
from app.nlp_parser import parse_prompt
from app.city_rates import resolve_city, get_cost_estimate, get_all_cities
from app import predictor

app = FastAPI(
    title="KSI Construction Estimator v3",
    description="""
## M/S. Khayti Steel Industries Limited — v3

**City-aware** ML-powered construction material + cost estimator.

### New in v3:
- **City/State based cost estimation** — 50+ Indian cities
- **Deep NLP** — auto-detects city from prompt ("in Chennai", "at Mumbai")
- **Full ₹ cost breakdown** — materials + labour
- **South Indian units** — supports cents, grounds
- **Cost per sqft** calculation

### Endpoints:
- `POST /estimate-from-prompt` → plain English description
- `POST /estimate`             → structured JSON
- `GET  /cities`               → all supported cities

### Example prompt:
```json
{"prompt": "3 BHK 3 floor residential house in Chennai"}
```
""",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup_event():
    predictor.load_model()
    predictor.load_meta()
    print("✅ Model loaded. KSI Estimator v3 ready.")


@app.get("/", tags=["Frontend"])
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"status": "ok", "version": "3.0.0", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health():
    meta = predictor.load_meta()
    return {
        "status":        "healthy",
        "version":       "3.0.0",
        "model":         "MultiOutputRegressor(RandomForestRegressor)",
        "train_samples": meta["train_samples"],
        "cities":        len(get_all_cities()),
    }


@app.get("/cities", tags=["Reference"])
def cities():
    """List all supported cities with their cost multipliers."""
    from app.city_rates import CITY_DB
    return {
        "total": len(CITY_DB),
        "cities": [
            {
                "city":        k.title(),
                "state":       v["state"],
                "tier":        v["tier"],
                "cost_mult":   v["cost_mult"],
                "cement_rate": v["cement"],
                "steel_rate":  v["steel"],
            }
            for k, v in sorted(CITY_DB.items(), key=lambda x: x[1]["state"])
        ]
    }


def _build_cost(materials: dict, city_input, total_sqft: float) -> CostEstimate:
    raw = get_cost_estimate(materials, city_input)
    raw["cost_per_sqft"] = round(raw["total_cost_inr"] / total_sqft, 0) if total_sqft else 0
    return CostEstimate(
        city=raw["city"],
        state=raw["state"],
        tier=raw["tier"],
        rates_used=raw["rates_used"],
        cost_breakdown=CostBreakdown(**raw["cost_breakdown"]),
        material_total=raw["material_total"],
        labour_cost=raw["labour_cost"],
        total_cost_inr=raw["total_cost_inr"],
        cost_per_sqft=raw["cost_per_sqft"],
    )


@app.post("/estimate", response_model=EstimateResponse,
          tags=["Estimation"], summary="Structured estimate with optional city")
def estimate(request: EstimateRequest):
    """
    Structured estimate. Add `city` field for local cost rates.

    ```json
    {
      "area": 1200, "unit": "sqft", "floors": 3,
      "building_type": 0, "quality": 1, "city": "Chennai"
    }
    ```
    """
    try:
        result = predictor.predict(
            area=request.area, unit=request.unit, floors=request.floors,
            building_type=int(request.building_type), quality=int(request.quality),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    cost = _build_cost(result["materials"], request.city, result["total_area_sqft"])

    return EstimateResponse(
        input_area_sqft=result["input_area_sqft"],
        total_area_sqft=result["total_area_sqft"],
        building_type=result["building_type"],
        quality=result["quality"],
        materials=MaterialQuantities(**result["materials"]),
        cost=cost,
        model_r2_scores=result["model_r2_scores"],
    )


@app.post("/estimate-from-prompt", response_model=PromptResponse,
          tags=["Estimation"], summary="Natural language estimate — city auto-detected")
def estimate_from_prompt(request: PromptRequest):
    """
    ## Plain English → Full Estimate with City Costs

    ### Examples:

    **Basic (city auto-detected):**
    ```json
    {"prompt": "3 BHK 3 floor residential house in Chennai"}
    ```

    **With area:**
    ```json
    {"prompt": "2BHK house 1200 sqft G+1 Coimbatore standard"}
    ```

    **South Indian units:**
    ```json
    {"prompt": "3 cents house coimbatore 2 floors standard"}
    ```

    **Premium commercial:**
    ```json
    {"prompt": "commercial 5 floors 250 sqm Mumbai premium"}
    ```

    **Industrial:**
    ```json
    {"prompt": "factory shed 5000 sqft 2 floors Pune"}
    ```

    **State level:**
    ```json
    {"prompt": "3 BHK 1500 sqft house Tamil Nadu G+1 economy"}
    ```
    """
    try:
        parsed = parse_prompt(request.prompt)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {e}")

    try:
        result = predictor.predict(
            area=parsed["area"], unit=parsed["unit"], floors=parsed["floors"],
            building_type=parsed["building_type"], quality=parsed["quality"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    BLBL = {0:"Residential",1:"Commercial",2:"Industrial"}
    QLBL = {0:"Economy",1:"Standard",2:"Premium"}

    cost = _build_cost(result["materials"], parsed.get("city_data", {}).get("city"),
                       result["total_area_sqft"])

    return PromptResponse(
        raw_prompt=parsed["raw_prompt"],
        parsed=ParsedDetails(
            area=parsed["area"], unit=parsed["unit"], floors=parsed["floors"],
            building_type=parsed["building_type"],
            building_label=BLBL[parsed["building_type"]],
            quality=parsed["quality"],
            quality_label=QLBL[parsed["quality"]],
            bhk=parsed["bhk"],
            city=parsed["city"],
            state=parsed["state"],
            parsed_notes=parsed["parsed_notes"],
        ),
        total_area_sqft=result["total_area_sqft"],
        materials=MaterialQuantities(**result["materials"]),
        cost=cost,
        model_r2_scores=result["model_r2_scores"],
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    return predictor.load_meta()