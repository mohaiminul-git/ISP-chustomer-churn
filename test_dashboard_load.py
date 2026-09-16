import time
import random
import io

import pandas as pd
import requests

from src.data_clean import data_cleaning

BASE_URL = "http://44.197.103.251:8080"
NUM_REQUESTS = 300

DELAY_SECONDS = 0.5
RAW_DATA_PATH = "data/raw/dataset.csv"


def build_single_payload(row: pd.Series) -> dict:
    return {
        "is_tv_subscriber": bool(row["is_tv_subscriber"]),
        "is_movie_package_subscriber": bool(row["is_movie_package_subscriber"]),
        "subscription_age": int(round(row["subscription_age"])),
        "bill_avg": int(round(row["bill_avg"])),
        "remaining_contract": float(row["reamining_contract"]) if pd.notna(row["reamining_contract"]) else 0.0,
        "service_failure_count": int(row["service_failure_count"]),
        "download_avg": float(row["download_avg"]) if pd.notna(row["download_avg"]) else 0.0,
        "upload_avg": float(row["upload_avg"]) if pd.notna(row["upload_avg"]) else 0.0,
        "download_over_limit": int(row["download_over_limit"]),
    }


def test_single_predictions(sample: pd.DataFrame):
    print(f"Sending {len(sample)} requests to /single_prediction ...")
    for _, row in sample.iterrows():
        payload = build_single_payload(row)
        resp = requests.post(f"{BASE_URL}/single_prediction", json=payload)
        print(resp.status_code, resp.json() if resp.ok else resp.text)
        time.sleep(DELAY_SECONDS)


def test_batch_prediction(sample: pd.DataFrame):
    print(f"Sending 1 batch request with {len(sample)} rows to /batch_prediction ...")
    cleaned = data_cleaning(sample.copy())
    csv_buffer = io.StringIO()
    cleaned.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    files = {"request": ("sample.csv", csv_buffer.getvalue(), "text/csv")}
    resp = requests.post(f"{BASE_URL}/batch_prediction", files=files)
    print(resp.status_code, resp.json() if resp.ok else resp.text)


def main():
    data = pd.read_csv(RAW_DATA_PATH)
    sample = data.sample(n=NUM_REQUESTS, random_state=random.randint(0, 10_000))

    test_single_predictions(sample)
    test_batch_prediction(sample)


if __name__ == "__main__":
    main()
