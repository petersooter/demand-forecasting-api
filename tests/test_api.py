"""
Tests for the demand forecasting API.
Runs against the FastAPI test client — no server required.

Usage:
  python tests/test_api.py
  pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Train model if not present
model_path = Path("model/demand_model.pkl")
if not model_path.exists():
    print("Training model for tests...")
    from scripts.train import train
    train(n_samples=3000, output_path=str(model_path))

from fastapi.testclient import TestClient
from app.main import app, predictor

predictor.load()
client = TestClient(app)

VALID_PAYLOAD = {
    "product_id": "SKU-0001",
    "store_id": "STORE-01",
    "year": 2026,
    "month": 6,
    "day_of_week": 2,
    "week_of_year": 24,
    "is_holiday": False,
    "is_weekend": False,
    "temperature_c": 21.5,
    "promotion_active": True,
    "price": 4.99,
    "lag_7_demand": 142.0,
    "lag_14_demand": 138.0,
    "rolling_28_avg": 145.3,
}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"
    assert isinstance(data["features"], list)


def test_forecast_valid():
    r = client.post("/forecast", json=VALID_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert "forecast_units" in data
    assert data["forecast_units"] >= 0
    assert data["lower_bound"] <= data["forecast_units"]
    assert data["upper_bound"] >= data["forecast_units"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_forecast_unknown_product():
    payload = {**VALID_PAYLOAD, "product_id": "SKU-UNKNOWN-99999"}
    r = client.post("/forecast", json=payload)
    assert r.status_code == 200
    assert r.json()["forecast_units"] >= 0


def test_forecast_weekend_effect():
    weekday = {**VALID_PAYLOAD, "is_weekend": False, "day_of_week": 2}
    weekend = {**VALID_PAYLOAD, "is_weekend": True,  "day_of_week": 6}
    wd = client.post("/forecast", json=weekday).json()["forecast_units"]
    we = client.post("/forecast", json=weekend).json()["forecast_units"]
    assert we >= wd * 0.8  # weekend should be >= weekday (model may vary)


def test_forecast_promo_lift():
    no_promo = {**VALID_PAYLOAD, "promotion_active": False}
    promo    = {**VALID_PAYLOAD, "promotion_active": True}
    base = client.post("/forecast", json=no_promo).json()["forecast_units"]
    lift = client.post("/forecast", json=promo).json()["forecast_units"]
    assert lift >= base * 0.9  # promo should generally lift demand


def test_forecast_invalid_month():
    payload = {**VALID_PAYLOAD, "month": 13}
    r = client.post("/forecast", json=payload)
    assert r.status_code == 422


def test_forecast_negative_price():
    payload = {**VALID_PAYLOAD, "price": -1.0}
    r = client.post("/forecast", json=payload)
    assert r.status_code == 422


def test_batch_forecast():
    r = client.post("/forecast/batch", json={"items": [VALID_PAYLOAD, VALID_PAYLOAD]})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["forecasts"]) == 2
    assert data["elapsed_seconds"] >= 0


def test_batch_size_limit():
    r = client.post("/forecast/batch", json={"items": [VALID_PAYLOAD] * 101})
    assert r.status_code == 400


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}  {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
