"""
Extended schemas — adds PromptRequest / PromptResponse for the /estimate-from-prompt endpoint.
Add these classes to schemas.py or import from here.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from app.schemas import MaterialQuantities


class PromptRequest(BaseModel):
    """
    Free-form natural language construction description.

    Examples
    --------
    "3 BHK 3 floor residential house"
    "2BHK house 1200 sqft G+1"
    "commercial building 5 floors 250 sqm premium"
    "small economy house 600 sqft"
    "4 BHK luxury villa G+2"
    "industrial warehouse 5000 sqft 2 floors"
    """
    prompt: str = Field(
        ...,
        min_length=3,
        description="Plain English description of the construction project.",
        examples=["3 BHK 3 floor residential house"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"prompt": "3 BHK 3 floor residential house"},
                {"prompt": "2BHK house 1200 sqft G+1 standard quality"},
                {"prompt": "commercial building 5 floors 250 sqm premium"},
            ]
        }
    }


class ParsedDetails(BaseModel):
    area:          float
    unit:          str
    floors:        int
    building_type: int
    building_label: str
    quality:       int
    quality_label: str
    bhk:           Optional[int]
    parsed_notes:  List[str]


class PromptResponse(BaseModel):
    raw_prompt:      str
    parsed:          ParsedDetails
    total_area_sqft: float
    materials:       MaterialQuantities
    model_r2_scores: Optional[dict] = None