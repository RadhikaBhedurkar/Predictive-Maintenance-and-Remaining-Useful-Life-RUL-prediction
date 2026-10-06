from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="Predictive Maintenance API",
    version="1.0.0",
    description="RUL prediction service using NASA C-MAPSS FD001."
)

try:
    pipeline = PredictionPipeline()
except Exception:
    pipeline = None

class MachineInput(BaseModel):
    setting_1: float = 0.0
    setting_2: float = 0.0
    setting_3: float = 100.0
    sensor_2: float = Field(..., examples=[642.0])
    sensor_3: float = Field(..., examples=[1589.0])
    sensor_4: float = Field(..., examples=[1400.0])
    sensor_7: float = Field(..., examples=[553.0])
    sensor_8: float = Field(..., examples=[2388.0])
    sensor_9: float = Field(..., examples=[9045.0])
    sensor_11: float = Field(..., examples=[47.0])
    sensor_12: float = Field(..., examples=[522.0])
    sensor_13: float = Field(..., examples=[2388.0])
    sensor_14: float = Field(..., examples=[8130.0])
    sensor_15: float = Field(..., examples=[8.5])
    sensor_17: float = Field(..., examples=[392.0])
    sensor_20: float = Field(..., examples=[39.0])
    sensor_21: float = Field(..., examples=[23.0])

@app.get("/health")
def health():
    return {"status": "ok" if pipeline else "model_not_loaded"}

@app.post("/predict")
def predict(data: MachineInput):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Train the model first."
        )

    rul = pipeline.predict(data.model_dump())

    if rul < 20:
        risk = "HIGH"
        recommendation = "Schedule maintenance immediately."
    elif rul < 50:
        risk = "MEDIUM"
        recommendation = "Plan maintenance soon and monitor the machine."
    else:
        risk = "LOW"
        recommendation = "Continue monitoring under normal maintenance policy."

    return {
        "predicted_RUL_cycles": round(rul, 2),
        "risk_level": risk,
        "recommendation": recommendation
    }
