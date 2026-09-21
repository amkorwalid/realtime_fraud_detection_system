import joblib
from pathlib import Path
import numpy as np


def load_models():
    path = Path.cwd() / "data_science"/ "models"
    path_xgb_model = Path(path, "xgb_model.pkl")
    path_lgbm_model = Path(path, "lgbm_model.pkl")

    loaded_xgb_model = joblib.load(path_xgb_model)
    loaded_lgbm_model = joblib.load(path_lgbm_model)

    return loaded_xgb_model, loaded_lgbm_model


def inference(xgb_model, lgbm_model, x, weight=0.80, threshold=0.95):

    xgb_probs = xgb_model.predict_proba(x)[:, 1]
    lgbm_probs = lgbm_model.predict_proba(x)[:, 1]

    ensemble_probs = weight * xgb_probs + (1 - weight) * lgbm_probs
    ensemble_preds = (ensemble_probs >= threshold).astype(int)

    return ensemble_preds