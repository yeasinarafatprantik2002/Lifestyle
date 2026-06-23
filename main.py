from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from life_style_train import MODEL_FILE
from life_style_train import main as train_lifestyle_model
from life_style_train import predict_from_input


BASE_DIR = Path(__file__).resolve().parent


def is_vercel_runtime():
    return os.getenv("VERCEL") == "1" or os.getenv("VERCEL_ENV") is not None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if is_vercel_runtime():
        app.state.training_status = "model_ready" if MODEL_FILE.exists() else "model_missing"
        app.state.training_result = {
            "runtime": "vercel",
            "model_file": str(MODEL_FILE),
            "message": "Vercel uses the pre-trained model artifact and does not train on cold start.",
        }
        yield
        return

    app.state.training_status = "training"
    try:
        app.state.training_result = train_lifestyle_model()
        app.state.training_status = "trained"
    except Exception as exc:
        app.state.training_status = "training_failed"
        app.state.training_result = {"error": str(exc)}
    yield


app = FastAPI(title="Lifestyle Health Risk Predictor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.state.training_status = "not_started"
app.state.training_result = None


class LifestyleInput(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: str
    height_cm: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)
    fruit_veg_freq: str
    fast_food_freq: str
    water_8_glasses: str
    sugary_drinks_freq: str
    exercise_freq: str
    exercise_type: Optional[str] = ""
    sleep_hours: str
    stress_level: str
    smoke: str
    alcohol: str
    chronic_conditions: Optional[str] = "None"
    family_history_heart: str
    fasting_glucose: Optional[str] = ""
    overall_health: str


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.post("/api/predict")
def predict(payload: LifestyleInput):
    if hasattr(payload, "model_dump"):
        return predict_from_input(payload.model_dump())
    return predict_from_input(payload.dict())


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "training_status": app.state.training_status,
        "training_result": app.state.training_result,
    }
