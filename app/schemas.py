"""
Pydantic schemas — v3 with city/state cost estimation
"""

from enum import IntEnum
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, field_validator


class BuildingType(IntEnum):
    residential = 0
    commercial  = 1
    industrial  = 2


class QualityGrade(IntEnum):
    economy  = 0
    standard = 1
    premium  = 2


# ── Structured Request ─────────────────────────────────────────────────────────
class EstimateRequest(BaseModel):
    area:          float        = Field(..., gt=0, examples=[1200.0])
    unit:          str          = Field("sqft", examples=["sqft", "sqm"])
    floors:        int          = Field(1, ge=1, le=15)
    building_type: BuildingType = Field(BuildingType.residential)
    quality:       QualityGrade = Field(QualityGrade.standard)
    city:          Optional[str]= Field(None, examples=["Chennai", "Mumbai"])

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, v):
        v = v.lower().strip()
        if v not in ("sqft", "sqm"):
            raise ValueError("unit must be 'sqft' or 'sqm'")
        return v

    model_config = {"json_schema_extra": {"examples": [
        {"area":1200,"unit":"sqft","floors":3,"building_type":0,"quality":1,"city":"Chennai"}
    ]}}


# ── NLP Prompt Request ─────────────────────────────────────────────────────────
class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=5,
        examples=["3 BHK 3 floor residential house in Chennai"])

    model_config = {"json_schema_extra": {"examples": [
        {"prompt": "3 BHK 3 floor residential house in Chennai"},
        {"prompt": "2BHK house 1200 sqft G+1 Coimbatore standard"},
        {"prompt": "commercial 5 floors 250 sqm Mumbai premium"},
        {"prompt": "4 BHK luxury villa G+2 Bangalore 2400 sqft"},
        {"prompt": "industrial warehouse 5000 sqft 2 floors Pune"},
        {"prompt": "3 cents house coimbatore 2 floors standard"},
    ]}}


# ── Sub-models ─────────────────────────────────────────────────────────────────
class MaterialQuantities(BaseModel):
    cement_bags:   int = Field(..., description="50 kg bags")
    sand_cft:      int = Field(..., description="Cubic feet")
    bricks:        int = Field(..., description="Nos")
    aggregate_cft: int = Field(..., description="Cubic feet")
    steel_kg:      int = Field(..., description="Kilograms")


class CostBreakdown(BaseModel):
    cement:    int
    sand:      int
    bricks:    int
    aggregate: int
    steel:     int


class CostEstimate(BaseModel):
    city:             str
    state:            str
    tier:             int
    rates_used:       Dict[str, float]
    cost_breakdown:   CostBreakdown
    material_total:   int
    labour_cost:      int
    total_cost_inr:   int
    cost_per_sqft:    float


class ParsedDetails(BaseModel):
    area:           float
    unit:           str
    floors:         int
    building_type:  int
    building_label: str
    quality:        int
    quality_label:  str
    bhk:            Optional[int] = None
    city:           str
    state:          str
    parsed_notes:   List[str]


# ── Responses ──────────────────────────────────────────────────────────────────
class EstimateResponse(BaseModel):
    input_area_sqft: float
    total_area_sqft: float
    building_type:   str
    quality:         str
    materials:       MaterialQuantities
    cost:            CostEstimate
    model_r2_scores: Optional[Dict[str, float]] = None


class PromptResponse(BaseModel):
    raw_prompt:      str
    parsed:          ParsedDetails
    total_area_sqft: float
    materials:       MaterialQuantities
    cost:            CostEstimate
    model_r2_scores: Optional[Dict[str, float]] = None