import numpy as np
import polars as pl
import joblib
from pathlib import Path


type_col_encoder_path = Path.cwd() / "data_science" / "models" / "type_col_encoder.pkl"
try:
    encoder = joblib.load(type_col_encoder_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Encoder file not found at {type_col_encoder_path}. Please ensure the encoder is trained and saved.")


def preprocess_data(data: dict):

    # load dataset
    df = pl.DataFrame(data)
    

    encoded_type = encoder.transform(df[["type"]])
    encoded_feature_names = encoder.get_feature_names_out(["type"])
    encoded_type = pl.DataFrame(encoded_type, schema=list(encoded_feature_names))
    df = pl.concat([df, encoded_type], how="horizontal").drop("type")

    df = df.drop("step", "nameOrig", "nameDest")

    int8_cols = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
    df = df.with_columns(
            [pl.col(col).cast(pl.Int8) for col in int8_cols]
        )
    df = df.with_columns(
            pl.col(pl.Float64).cast(pl.Float32)
        )
    
    return df
