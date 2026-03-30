"""
Pydantic schemas for request / response validation
"""

from enum import IntEnum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class BuildingType(IntEnum):
    residential = 0
    commercial  = 1
    industrial  = 2


class QualityGrade(IntEnum):
    economy  = 0
    standard = 1
    premium  = 2


class EstimateRequest(BaseModel):
    """
    Prompting guide
    ───────────────
    area          : Built-up area per floor (numeric > 0)
    unit          : "sqft" or "sqm"
    floors        : 1–10
    building_type : 0=Residential | 1=Commercial | 2=Industrial   (default 0)
    quality       : 0=Economy     | 1=Standard   | 2=Premium      (default 1)
    """
    area: float = Field(
        ...,
        gt=0,
        description="Built-up area of the project (must be > 0).",
        examples=[1200.0],
    )
    unit: str = Field(
        "sqft",
        description="Unit of area: 'sqft' or 'sqm'.",
        examples=["sqft", "sqm"],
    )
    floors: int = Field(
        1,
        ge=1,
        le=10,
        description="Number of floors (1–10).",
        examples=[2],
    )
    building_type: BuildingType = Field(
        BuildingType.residential,
        description="0=Residential, 1=Commercial, 2=Industrial.",
    )
    quality: QualityGrade = Field(
        QualityGrade.standard,
        description="0=Economy, 1=Standard, 2=Premium.",
    )

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("sqft", "sqm"):
            raise ValueError("unit must be 'sqft' or 'sqm'")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "area": 1200,
                    "unit": "sqft",
                    "floors": 2,
                    "building_type": 0,
                    "quality": 1,
                }
            ]
        }
    }


class MaterialQuantities(BaseModel):
    cement_bags:   int = Field(..., description="Number of 50 kg cement bags.")
    sand_cft:      int = Field(..., description="Sand in cubic feet.")
    bricks:        int = Field(..., description="Number of bricks.")
    aggregate_cft: int = Field(..., description="Stone aggregate in cubic feet.")
    steel_kg:      int = Field(..., description="Reinforcement steel in kg.")


class EstimateResponse(BaseModel):
    input_area_sqft: float
    total_area_sqft: float
    building_type:   str
    quality:         str
    materials:       MaterialQuantities
    model_r2_scores: Optional[dict] = None