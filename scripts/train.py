"""
Train the demand forecasting model and save to model/demand_model.pkl.

Uses a GradientBoostingRegressor on synthetic retail demand data.
In production, replace generate_training_data() with your actual dataset.

Usage:
  python scripts/train.py
  python scripts/train.py --samples 20000 --output model/demand_model.pkl
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    "product_enc", "store_enc", "year", "month", "day_of_week",
    "week_of_year", "is_holiday", "is_weekend", "temperature_c",
    "promotion_active", "price", "lag_7_demand", "lag_14_demand", "rolling_28_avg",
]

MODEL_VERSION = "1.0.0"


def generate_training_data(n_samples: int = 15000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic retail demand data for training.

    Demand is a function of:
      - Base product/store demand level
      - Seasonal pattern (month + week)
      - Day-of-week effect (weekends higher)
      - Temperature effect (higher temps boost certain products)
      - Promotion lift (20-40% uplift)
      - Price elasticity (higher price = lower demand)
      - Lag demand (autoregressive component)
    """
    rng = np.random.default_rng(seed)

    products = [f"SKU-{i:04d}" for i in range(1, 51)]
    stores   = [f"STORE-{i:02d}" for i in range(1, 21)]

    product_base = {p: rng.integers(80, 300) for p in products}
    store_mult   = {s: rng.uniform(0.7, 1.4) for s in stores}

    rows = []
    for _ in range(n_samples):
        product   = rng.choice(products)
        store     = rng.choice(stores)
        year      = int(rng.integers(2023, 2027))
        month     = int(rng.integers(1, 13))
        dow       = int(rng.integers(0, 7))
        woy       = int(rng.integers(1, 54))
        is_hol    = bool(rng.random() < 0.05)
        is_wknd   = dow >= 5
        temp      = float(rng.uniform(-5, 38))
        promo     = bool(rng.random() < 0.25)
        price     = float(round(rng.uniform(1.5, 25.0), 2))
        lag7      = float(rng.integers(50, 350))
        lag14     = float(lag7 * rng.uniform(0.85, 1.15))
        rolling28 = float(lag7 * rng.uniform(0.9, 1.1))

        # Demand model
        base     = product_base[product] * store_mult[store]
        seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * month / 12)
        wknd     = 1.25 if is_wknd else 1.0
        hol      = 1.35 if is_hol else 1.0
        tmp      = 1.0 + 0.005 * max(0, temp - 15)
        promo_fx = rng.uniform(1.20, 1.40) if promo else 1.0
        price_fx = max(0.5, 1.0 - 0.02 * (price - 5))
        ar       = 0.4 * lag7 / base if base > 0 else 0
        noise    = rng.normal(0, 0.05)

        demand = base * seasonal * wknd * hol * tmp * promo_fx * price_fx * (1 + ar + noise)
        demand = max(0, round(demand))

        rows.append({
            "product_id": product, "store_id": store,
            "year": year, "month": month, "day_of_week": dow,
            "week_of_year": woy, "is_holiday": int(is_hol),
            "is_weekend": int(is_wknd), "temperature_c": temp,
            "promotion_active": int(promo), "price": price,
            "lag_7_demand": lag7, "lag_14_demand": lag14,
            "rolling_28_avg": rolling28, "demand": demand,
        })

    return pd.DataFrame(rows)


def train(n_samples: int = 15000, output_path: str = "model/demand_model.pkl") -> None:
    print(f"Generating {n_samples:,} training samples...")
    df = generate_training_data(n_samples)

    # Encode categoricals
    prod_enc  = LabelEncoder().fit(df["product_id"])
    store_enc = LabelEncoder().fit(df["store_id"])
    df["product_enc"] = prod_enc.transform(df["product_id"])
    df["store_enc"]   = store_enc.transform(df["store_id"])

    X = df[FEATURE_COLS]
    y = df["demand"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = mean_squared_error(y_test, y_pred) ** 0.5
    mape   = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100)

    print(f"\nTest set metrics:")
    print(f"  MAE:  {mae:.2f} units")
    print(f"  RMSE: {rmse:.2f} units")
    print(f"  MAPE: {mape:.1f}%")

    # Residual std for confidence intervals
    residuals    = y_test - y_pred
    residual_std = float(residuals.std())

    # Save model bundle
    bundle = {
        "model":        model,
        "product_enc":  prod_enc,
        "store_enc":    store_enc,
        "feature_cols": FEATURE_COLS,
        "residual_std": residual_std,
        "version":      MODEL_VERSION,
        "trained_at":   datetime.utcnow().isoformat() + "Z",
        "metrics":      {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 1)},
        "n_samples":    n_samples,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(bundle, f)

    # Save metrics separately for CI/CD
    metrics_path = out.parent / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(bundle["metrics"], f, indent=2)

    print(f"\nModel saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15000)
    parser.add_argument("--output",  default="model/demand_model.pkl")
    args = parser.parse_args()
    train(args.samples, args.output)
