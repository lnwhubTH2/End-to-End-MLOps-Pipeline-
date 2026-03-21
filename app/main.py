from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Iris Model API")

# Dynamically locate the newly promoted Champion model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "champion_model.pkl")

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Iris Model loaded successfully.")
    else:
        print("Model file not found. Running in mock mode.")
except Exception as e:
    print(f"Error loading model: {e}. Running in mock mode.")

class IrisFeature(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction_class: int
    confidence: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Model API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(feature: IrisFeature):
    if model is None:
        return {"prediction_class": 0, "confidence": 0.99}
    
    try:
        # Prepare feature vector from JSON inputs
        input_data = pd.DataFrame([feature.model_dump()])
        
        # Predict class
        pred_class = model.predict(input_data)[0]
        
        # DecisionTreeClassifier predicts probabilities
        probs = model.predict_proba(input_data)[0]
        confidence = max(probs)
        
        return {"prediction_class": int(pred_class), "confidence": float(confidence)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
