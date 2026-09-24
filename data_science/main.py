import json
from pydantic import BaseModel
from fastapi import FastAPI
from pathlib import Path
from inference import load_models, inference_model, inference_model_ensemble
from preprocess import preprocess_data

# 1. Initialize the FastAPI instance
app = FastAPI()

class Data(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float

class InferenceResult(BaseModel):
    # xgb_result: float
    lgbm_result: list[int]


xgb_model, lgbm_model = load_models()


@app.get("/")
def read_root():
    
    return {"message": "Hello World"}


@app.post("/api/inference")
def inference(data: list[Data]) -> InferenceResult:

    # empty
    if not data:
        return InferenceResult(
            # xgb_result=0.0,
            lgbm_result=[]
        )

    data_preprocessed = preprocess_data(data)

    # Perform inference
    # xgb_result = inference_model(xgb_model, data_preprocessed)[0]
    lgbm_result = inference_model(lgbm_model, data_preprocessed)

    return InferenceResult(
        # xgb_result=xgb_result,
        lgbm_result=lgbm_result
    )