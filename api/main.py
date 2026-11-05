"""REST API for the concrete strength model.

Run from the project folder:   uvicorn api.main:app --reload
Interactive docs (Swagger UI): http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import InputValidationError, load_artifacts, predict_strength

app = FastAPI(
    title="Concrete Compressive Strength API",
    description="Predicts concrete compressive strength (MPa) from mix proportions (kg/m³) and curing age (days).",
    version="1.0.0",
)
model, meta = load_artifacts()


class MixInput(BaseModel):
    cement: float = Field(..., allow_inf_nan=False, examples=[280.0], description="Cement, kg/m³")
    slag: float = Field(..., allow_inf_nan=False, examples=[0.0], description="Blast furnace slag, kg/m³")
    fly_ash: float = Field(..., allow_inf_nan=False, examples=[0.0], description="Fly ash, kg/m³")
    water: float = Field(..., allow_inf_nan=False, examples=[190.0], description="Water, kg/m³")
    superplasticizer: float = Field(..., allow_inf_nan=False, examples=[0.0], description="Superplasticizer, kg/m³")
    coarse_agg: float = Field(..., allow_inf_nan=False, examples=[970.0], description="Coarse aggregate, kg/m³")
    fine_agg: float = Field(..., allow_inf_nan=False, examples=[800.0], description="Fine aggregate, kg/m³")
    age: float = Field(..., allow_inf_nan=False, examples=[28], description="Curing age, days")


class Prediction(BaseModel):
    strength_mpa: float
    low_mpa: float
    high_mpa: float
    category: str
    water_cement: float
    water_binder: float
    warnings: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["model_name"]}


@app.get("/model-info")
def model_info():
    return {"model": meta["model_name"], "test_metrics": meta["test_metrics"],
            "feature_ranges": meta["feature_ranges"], "dataset": meta["dataset"]}


@app.post("/predict", response_model=Prediction)
def predict(mix: MixInput):
    try:
        result = predict_strength(model, meta, mix.model_dump())
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors) from exc
    return {k: result[k] for k in Prediction.model_fields}
