"""
demand-forecasting-api
FastAPI service that returns demand forecasts from a trained scikit-learn model.

Endpoints:
  GET  /              Health check
  GET  /health        Detailed health + model metadata
  POST /forecast      Single forecast
  POST /forecast/batch  Batch forecast (up to 100 items)
  GET  /docs          Auto-generated Swagger UI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from .schemas import ForecastRequest, ForecastResponse, BatchForecastRequest, BatchForecastResponse, HealthResponse
from .predictor import Predictor

predictor = Predictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(
    title="Demand Forecasting API",
    description=(
        "Demand forecasting service built on a scikit-learn gradient boosting model. "
        "Accepts product, store, and temporal features and returns point forecasts "
        "with confidence intervals. Built as a portfolio demo by Peter Sooter."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "demand-forecasting-api", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version,
        features=predictor.feature_names,
        trained_at=predictor.trained_at,
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
def forecast(request: ForecastRequest):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predictor.predict_one(request)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


@app.post("/forecast/batch", response_model=BatchForecastResponse, tags=["Forecast"])
def forecast_batch(request: BatchForecastRequest):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(request.items) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit is 100 items")
    try:
        t0 = time.perf_counter()
        results = [predictor.predict_one(item) for item in request.items]
        elapsed = round(time.perf_counter() - t0, 3)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    return BatchForecastResponse(forecasts=results, count=len(results), elapsed_seconds=elapsed)
