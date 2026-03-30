"""
Construction Material Estimator API
=====================================
FastAPI app — ML-powered material quantity predictions.

Endpoints:
  POST /estimate              — structured JSON input
  POST /estimate-from-prompt  — plain English input
  GET  /health
  GET  /model/info

Run  : uvicorn app.main:app --reload
Docs : http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.schemas import EstimateRequest, EstimateResponse, MaterialQuantities
from app.prompt_schemas import PromptRequest, PromptResponse, ParsedDetails
from app.nlp_parser import parse_prompt
from app import predictor

app = FastAPI(
    title="Construction Material Estimator",
    description="""
## M/S. Khayti Steel Industries Limited

ML-powered construction material quantity estimator.

### Two ways to call:

**1. Structured input** → `POST /estimate`
```json
{"area": 1200, "unit": "sqft", "floors": 3, "building_type": 0, "quality": 1}
```

**2. Plain English prompt** → `POST /estimate-from-prompt`
```json
{"prompt": "3 BHK 3 floor residential house"}
```

**building_type:** 0=Residential | 1=Commercial | 2=Industrial

**quality:** 0=Economy | 1=Standard | 2=Premium
""",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static Files & Templates ──────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup_event():
    predictor.load_model()
    predictor.load_meta()
    print("Model loaded and ready.")


@app.get("/", tags=["Frontend"])
def root():
    """Return the main frontend SPA."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"status": "ok", "message": "Static frontend not found yet. Please wait."}


@app.get("/health", tags=["Health"])
def health():
    meta = predictor.load_meta()
    return {
        "status":        "healthy",
        "model":         "MultiOutputRegressor(RandomForestRegressor)",
        "train_samples": meta["train_samples"],
        "targets":       meta["targets"],
    }


@app.post("/estimate", response_model=EstimateResponse,
          tags=["Estimation — structured"],
          summary="Estimate via structured JSON fields")
def estimate(request: EstimateRequest):
    """Submit structured fields and get material quantities."""
    try:
        result = predictor.predict(
            area=request.area,
            unit=request.unit,
            floors=request.floors,
            building_type=int(request.building_type),
            quality=int(request.quality),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return EstimateResponse(
        input_area_sqft=result["input_area_sqft"],
        total_area_sqft=result["total_area_sqft"],
        building_type=result["building_type"],
        quality=result["quality"],
        materials=MaterialQuantities(**result["materials"]),
        model_r2_scores=result["model_r2_scores"],
    )


@app.post("/estimate-from-prompt", response_model=PromptResponse,
          tags=["Estimation — NLP prompt"],
          summary="Estimate via plain English description")
def estimate_from_prompt(request: PromptRequest):
    """
    ## Estimate from a Natural Language Prompt

    Describe your project in plain English — the API extracts details automatically.

    ### Prompt examples

    **3 BHK house (your exact use case):**
    ```json
    {"prompt": "3 BHK 3 floor residential house"}
    ```

    **With explicit area:**
    ```json
    {"prompt": "2BHK house 1200 sqft G+1 standard quality"}
    ```

    **Commercial in sqm:**
    ```json
    {"prompt": "commercial building 5 floors 250 sqm premium"}
    ```

    **Economy home:**
    ```json
    {"prompt": "small economy house 600 sqft"}
    ```

    **Luxury villa:**
    ```json
    {"prompt": "4 BHK luxury villa G+2"}
    ```

    **Industrial:**
    ```json
    {"prompt": "industrial warehouse 5000 sqft 2 floors"}
    ```

    ### What gets auto-detected:
    - BHK count → area per floor (1BHK=500, 2BHK=850, 3BHK=1200, 4BHK=1800 sqft)
    - Number of floors (1-10, G+N notation)
    - Area + unit (sqft / sqm)
    - Building type (residential / commercial / industrial)
    - Quality grade (economy / standard / premium)
    """
    try:
        parsed = parse_prompt(request.prompt)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parse error: {str(e)}")

    try:
        result = predictor.predict(
            area=parsed["area"],
            unit=parsed["unit"],
            floors=parsed["floors"],
            building_type=parsed["building_type"],
            quality=parsed["quality"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    BUILDING_LABELS = {0: "Residential", 1: "Commercial", 2: "Industrial"}
    QUALITY_LABELS  = {0: "Economy",     1: "Standard",   2: "Premium"}

    return PromptResponse(
        raw_prompt=parsed["raw_prompt"],
        parsed=ParsedDetails(
            area=parsed["area"],
            unit=parsed["unit"],
            floors=parsed["floors"],
            building_type=parsed["building_type"],
            building_label=BUILDING_LABELS[parsed["building_type"]],
            quality=parsed["quality"],
            quality_label=QUALITY_LABELS[parsed["quality"]],
            bhk=parsed["bhk"],
            parsed_notes=parsed["parsed_notes"],
        ),
        total_area_sqft=result["total_area_sqft"],
        materials=MaterialQuantities(**result["materials"]),
        model_r2_scores=result["model_r2_scores"],
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns training metadata and R2 / MAE scores per target."""
    return predictor.load_meta()