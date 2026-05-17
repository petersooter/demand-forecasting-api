"""
Request and response schemas.
All inputs validated by Pydantic before reaching the predictor.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class ForecastRequest(BaseModel):
    """
    Input features for a single demand forecast.

    Example:
      {
        "product_id": "SKU-1042",
        "store_id": "STORE-07",
        "year": 2026,
        "month": 6,
        "day_of_week": 2,
        "week_of_year": 24,
        "is_holiday": false,
        "is_weekend": false,
        "temperature_c": 21.5,
        "promotion_active": true,
        "price": 4.99,
        "lag_7_demand": 142,
        "lag_14_demand": 138,
        "rolling_28_avg": 145.3
      }
    """
    product_id:       str   = Field(..., example="SKU-1042")
    store_id:         str   = Field(..., example="STORE-07")
    year:             int   = Field(..., ge=2020, le=2035, example=2026)
    month:            int   = Field(..., ge=1, le=12, example=6)
    day_of_week:      int   = Field(..., ge=0, le=6, example=2, description="0=Monday, 6=Sunday")
    week_of_year:     int   = Field(..., ge=1, le=53, example=24)
    is_holiday:       bool  = Field(False, example=False)
    is_weekend:       bool  = Field(False, example=False)
    temperature_c:    float = Field(..., ge=-40.0, le=60.0, example=21.5)
    promotion_active: bool  = Field(False, example=True)
    price:            float = Field(..., gt=0, example=4.99)
    lag_7_demand:     float = Field(..., ge=0, example=142.0, description="Actual demand 7 days ago")
    lag_14_demand:    float = Field(..., ge=0, example=138.0, description="Actual demand 14 days ago")
    rolling_28_avg:   float = Field(..., ge=0, example=145.3, description="Rolling 28-day average demand")

    model_config = {"json_schema_extra": {"example": {
        "product_id": "SKU-1042", "store_id": "STORE-07",
        "year": 2026, "month": 6, "day_of_week": 2, "week_of_year": 24,
        "is_holiday": False, "is_weekend": False,
        "temperature_c": 21.5, "promotion_active": True,
        "price": 4.99, "lag_7_demand": 142.0,
        "lag_14_demand": 138.0, "rolling_28_avg": 145.3,
    }}}


class ForecastResponse(BaseModel):
    product_id:         str
    store_id:           str
    forecast_units:     float = Field(..., description="Point forecast — expected unit demand")
    lower_bound:        float = Field(..., description="80% confidence interval lower bound")
    upper_bound:        float = Field(..., description="80% confidence interval upper bound")
    confidence:         float = Field(..., description="Model confidence score (0-1)")
    model_version:      str


class BatchForecastRequest(BaseModel):
    items: List[ForecastRequest] = Field(..., min_length=1)


class BatchForecastResponse(BaseModel):
    forecasts:       List[ForecastResponse]
    count:           int
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: Optional[str]
    features:      Optional[List[str]]
    trained_at:    Optional[str]
