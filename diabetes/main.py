from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

diabet_app = FastAPI(title='Predict Diabetes')


class Diabet(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / 'diabet_model.pkl'
scaler_path = BASE_DIR / 'scaler_diabet.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


@diabet_app.post('/predict/')
async def predict_diabet(diabet: Diabet):
    diabet_dict = list(diabet.dict().values())
    scaled = scaler.transform([diabet_dict])
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    return {'approved': bool(pred), 'probability': round(prob, 2)}








