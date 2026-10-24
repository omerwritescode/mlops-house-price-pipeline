"""Request/response schema for the prediction API (pydantic v2).

Bounds mirror configs/config.yaml -> validation.ranges so the API rejects the same
impossible inputs the data gate rejects. The three columns that are missing in the
supplied data may be omitted; the model imputes them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Listing(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_extra={"example": {
        "Area_sqft": 1150, "Bedrooms": 2, "Bathrooms": 2, "Age_Years": 12, "Distance_to_Center_km": 9.5,
        "Neighborhood_Score": 6.5, "Has_Parking": 1, "Property_Condition": "Good", "City_Zone": "Central"}})

    Area_sqft: float = Field(ge=100, le=10000, description="Built-up area in square feet")
    Bedrooms: int = Field(ge=1, le=10)
    Bathrooms: int = Field(ge=1, le=10)
    Age_Years: float | None = Field(default=None, ge=0, le=100)
    Distance_to_Center_km: float | None = Field(default=None, ge=0, le=100)
    Neighborhood_Score: float | None = Field(default=None, ge=1, le=10)
    Has_Parking: Literal[0, 1]
    Property_Condition: Literal["Poor", "Fair", "Good", "Excellent"]
    City_Zone: Literal["North", "South", "East", "West", "Central"]


class Prediction(BaseModel):
    predicted_price_lakhs: float
    model_version: str


class BatchRequest(BaseModel):
    listings: list[Listing] = Field(min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    predictions: list[float]
    model_version: str
