"""Send rows from a CSV file to the deployed endpoint and print the predictions.

    python azure/predict.py data/new_patients.csv

The CSV needs the same 30 feature columns the model was trained on (no target column).
"""
import sys

import pandas as pd
import requests

from common import ENDPOINT_NAME, get_client

LABELS = {0: "malignant", 1: "benign"}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python azure/predict.py <file.csv>")
    df = pd.read_csv(sys.argv[1])

    ml_client = get_client()
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    key = ml_client.online_endpoints.get_keys(ENDPOINT_NAME).primary_key

    payload = {"input_data": {
        "columns": list(df.columns),
        "index": list(range(len(df))),
        "data": df.values.tolist(),
    }}
    r = requests.post(
        endpoint.scoring_uri,
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
        timeout=60,
    )
    if not r.ok:
        raise SystemExit(f"Endpoint returned {r.status_code}: {r.text}")

    df.insert(0, "prediction", [LABELS.get(p, p) for p in r.json()])
    print(df[["prediction", "mean radius", "mean texture", "mean area"]].to_string())


if __name__ == "__main__":
    main()
