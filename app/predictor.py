"""
Predictor: loads the trained model bundle and serves predictions.
Handles label encoding for unseen product/store IDs gracefully.
"""

import pickle
from pathlib import Path
from typing import Optional
import numpy as np

MODEL_PATH = Path(__file__).parent.parent / "model" / "demand_model.pkl"
CONFIDENCE_INTERVAL_Z = 1.28   # 80% CI


class Predictor:
    def __init__(self):
        self.model        = None
        self.product_enc  = None
        self.store_enc    = None
        self.feature_cols = None
        self.residual_std = 0.0
        self.model_version = None
        self.trained_at    = None
        self.is_loaded     = False

    def load(self, path: Optional[Path] = None) -> None:
        path = path or MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run: python scripts/train.py"
            )
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self.model        = bundle["model"]
        self.product_enc  = bundle["product_enc"]
        self.store_enc    = bundle["store_enc"]
        self.feature_cols = bundle["feature_cols"]
        self.residual_std = bundle.get("residual_std", 20.0)
        self.model_version = bundle.get("version", "unknown")
        self.trained_at    = bundle.get("trained_at")
        self.is_loaded     = True

    @property
    def feature_names(self):
        return self.feature_cols if self.is_loaded else None

    def _encode(self, encoder, value: str) -> int:
        """Encode a categorical. Returns 0 for unseen values (graceful fallback)."""
        classes = list(encoder.classes_)
        if value in classes:
            return int(encoder.transform([value])[0])
        return 0

    def predict_one(self, request) -> dict:
        """
        Run inference on a single ForecastRequest.
        Returns a dict matching ForecastResponse schema.
        """
        features = {
            "product_enc":    self._encode(self.product_enc, request.product_id),
            "store_enc":      self._encode(self.store_enc,   request.store_id),
            "year":           request.year,
            "month":          request.month,
            "day_of_week":    request.day_of_week,
            "week_of_year":   request.week_of_year,
            "is_holiday":     int(request.is_holiday),
            "is_weekend":     int(request.is_weekend),
            "temperature_c":  request.temperature_c,
            "promotion_active": int(request.promotion_active),
            "price":          request.price,
            "lag_7_demand":   request.lag_7_demand,
            "lag_14_demand":  request.lag_14_demand,
            "rolling_28_avg": request.rolling_28_avg,
        }

        import pandas as pd
        X = pd.DataFrame([features], columns=self.feature_cols)
        point = float(self.model.predict(X)[0])
        point = max(0.0, round(point, 1))

        margin = CONFIDENCE_INTERVAL_Z * self.residual_std
        lower  = max(0.0, round(point - margin, 1))
        upper  = round(point + margin, 1)

        # Confidence: inverse of relative uncertainty, capped 0-1
        rel_uncertainty = (margin / point) if point > 0 else 1.0
        confidence = round(max(0.0, min(1.0, 1.0 - rel_uncertainty * 0.5)), 3)

        return {
            "product_id":    request.product_id,
            "store_id":      request.store_id,
            "forecast_units": point,
            "lower_bound":   lower,
            "upper_bound":   upper,
            "confidence":    confidence,
            "model_version": self.model_version,
        }
