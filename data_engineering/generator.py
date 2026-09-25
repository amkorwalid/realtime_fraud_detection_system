import numpy as np
from pathlib import Path
import pandas as pd
from json import loads

def data_loader():
    path = Path(__file__).parent.parent / "data" / "stream" / "stream.csv"
    df = pd.read_csv(path)
    result = df["isFraud"]
    df = df.drop(columns=["isFraud", "isFlaggedFraud"])
    df = loads(df.to_json(orient="records"))
    result = loads(result.to_json(orient="records"))
    return df, result

def generate_request_schedule(total_requests: int, duration: float):
    
    num_buckets = duration 
    time_points = np.linspace(0, duration, num_buckets)

    
    mean = duration / 2.0 # 
    std_dev = duration / 4.0

    # Probability density function (PDF) for normal distribution
    pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((time_points - mean) / std_dev) ** 2
    )

    # Normalize probabilities to sum to 1
    probabilities = pdf / np.sum(pdf)

    # Scale to total target requests and round to discrete integer counts
    requests_per_bucket = np.round(probabilities * total_requests).astype(int)

    # Adjust rounding discrepancy to guarantee exact total count
    discrepancy = total_requests - np.sum(requests_per_bucket)
    requests_per_bucket[num_buckets // 2] += discrepancy

    return requests_per_bucket


