from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from fastapi import File, UploadFile
from io import StringIO


# ============================================================
# Configuration
# ============================================================
MODEL_DIR = "model_store"
MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI(
    title="Traditional Anomaly Detection API",
    version="1.0.0",
    description="FastAPI framework for anomaly detection using Isolation Forest"
)


# ============================================================
# Request / Response Schemas
# ============================================================
class TrainRequest(BaseModel):
    records: List[Dict[str, float]] = Field(
        ...,
        description="Training data as list of numeric feature dictionaries"
    )
    contamination: float = Field(
        0.05,
        ge=0.001,
        le=0.5,
        description="Expected proportion of anomalies in the data"
    )
    random_state: int = 42
    n_estimators: int = 200


class PredictRequest(BaseModel):
    records: List[Dict[str, float]] = Field(
        ...,
        description="Prediction data as list of numeric feature dictionaries"
    )


class PredictionItem(BaseModel):
    input: Dict[str, float]
    anomaly_score: float
    prediction: int
    label: str


class PredictResponse(BaseModel):
    predictions: List[PredictionItem]


class ModelInfoResponse(BaseModel):
    model_available: bool
    algorithm: Optional[str] = None
    features: Optional[List[str]] = None
    contamination: Optional[float] = None
    n_estimators: Optional[int] = None

class TrainCsvRequest(BaseModel):
    csv_path: str = Field(..., description="Full path of the CSV file")
    contamination: float = Field(
        0.05,
        ge=0.001,
        le=0.5,
        description="Expected proportion of anomalies in the data"
    )
    random_state: int = 42
    n_estimators: int = 200
    
# ============================================================
# Utility functions
# ============================================================
def save_model(model: Pipeline, metadata: Dict[str, Any]) -> None:
    joblib.dump(model, MODEL_PATH)
    joblib.dump(metadata, METADATA_PATH)


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


def validate_and_prepare_dataframe(records: List[Dict[str, float]], expected_features: Optional[List[str]] = None) -> pd.DataFrame:
    if not records:
        raise ValueError("Input records cannot be empty.")

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("Could not create dataframe from input records.")

    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if expected_features is not None:
        missing_cols = [col for col in expected_features if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in expected_features]

        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")
        if extra_cols:
            raise ValueError(f"Unexpected extra feature columns: {extra_cols}")

        df = df[expected_features]

    return df


def build_pipeline(contamination: float, random_state: int, n_estimators: int) -> Pipeline:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators
        ))
    ])
    return pipeline


# ============================================================
# API endpoints
# ============================================================
@app.get("/health")
def health_check():
    model_exists = os.path.exists(MODEL_PATH)
    return {
        "status": "ok",
        "model_loaded": model_exists
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    model, metadata = load_model()

    if model is None or metadata is None:
        return ModelInfoResponse(model_available=False)

    return ModelInfoResponse(
        model_available=True,
        algorithm=metadata.get("algorithm"),
        features=metadata.get("features"),
        contamination=metadata.get("contamination"),
        n_estimators=metadata.get("n_estimators")
    )


@app.post("/train")
def train_model(request: TrainRequest):
    try:
        df = validate_and_prepare_dataframe(request.records)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if df.shape[0] < 10:
        raise HTTPException(
            status_code=400,
            detail="At least 10 records are recommended for training."
        )

    pipeline = build_pipeline(
        contamination=request.contamination,
        random_state=request.random_state,
        n_estimators=request.n_estimators
    )

    pipeline.fit(df)

    metadata = {
        "algorithm": "IsolationForest",
        "features": df.columns.tolist(),
        "contamination": request.contamination,
        "n_estimators": request.n_estimators
    }

    save_model(pipeline, metadata)

    return {
        "message": "Model trained successfully.",
        "num_records": len(df),
        "num_features": len(df.columns),
        "features": df.columns.tolist(),
        "contamination": request.contamination
    }

@app.post("/train-upload-csv")
async def train_model_upload_csv(
    file: UploadFile = File(...),
    contamination: float = 0.05,
    random_state: int = 42,
    n_estimators: int = 200
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read uploaded CSV: {str(e)}"
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    df = df.select_dtypes(include=[np.number])

    if df.empty:
        raise HTTPException(status_code=400, detail="No numeric columns found in CSV.")

    if df.shape[0] < 10:
        raise HTTPException(status_code=400, detail="At least 10 records are recommended for training.")

    pipeline = build_pipeline(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators
    )

    pipeline.fit(df)

    metadata = {
        "algorithm": "IsolationForest",
        "features": df.columns.tolist(),
        "contamination": contamination,
        "n_estimators": n_estimators,
        "source": file.filename
    }

    save_model(pipeline, metadata)

    return {
        "message": "Model trained successfully from uploaded CSV.",
        "file_name": file.filename,
        "num_records": len(df),
        "num_features": len(df.columns),
        "features": df.columns.tolist(),
        "model_path": MODEL_PATH
    }
    
@app.post("/train-from-csv")
def train_model_from_csv(request: TrainCsvRequest):
    if not os.path.exists(request.csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"CSV file not found: {request.csv_path}"
        )

    try:
        df = pd.read_csv(request.csv_path)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read CSV file: {str(e)}"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty."
        )

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="No numeric columns found in CSV."
        )

    if df.shape[0] < 10:
        raise HTTPException(
            status_code=400,
            detail="At least 10 records are recommended for training."
        )

    pipeline = build_pipeline(
        contamination=request.contamination,
        random_state=request.random_state,
        n_estimators=request.n_estimators
    )

    pipeline.fit(df)

    metadata = {
        "algorithm": "IsolationForest",
        "features": df.columns.tolist(),
        "contamination": request.contamination,
        "n_estimators": request.n_estimators,
        "source": request.csv_path
    }

    save_model(pipeline, metadata)

    return {
        "message": "Model trained successfully from CSV.",
        "csv_path": request.csv_path,
        "num_records": len(df),
        "num_features": len(df.columns),
        "features": df.columns.tolist(),
        "model_path": MODEL_PATH,
        "metadata_path": METADATA_PATH
    }
    
@app.post("/predict", response_model=PredictResponse)
def predict_anomaly(request: PredictRequest):
    model, metadata = load_model()

    if model is None or metadata is None:
        raise HTTPException(
            status_code=400,
            detail="Model not found. Train the model first using /train."
        )

    expected_features = metadata["features"]

    try:
        df = validate_and_prepare_dataframe(request.records, expected_features=expected_features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # IsolationForest:
    # predict => 1 means normal, -1 means anomaly
    raw_predictions = model.predict(df)
    raw_scores = model.decision_function(df)

    results = []
    for record, score, pred in zip(request.records, raw_scores, raw_predictions):
        label = "anomaly" if pred == -1 else "normal"
        results.append(
            PredictionItem(
                input=record,
                anomaly_score=float(score),
                prediction=int(pred),
                label=label
            )
        )

    return PredictResponse(predictions=results)
