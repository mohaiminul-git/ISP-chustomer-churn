from fastapi import FastAPI
from fastapi.testclient import TestClient
from api import app
from api.lifespan import get_model

VALID_PAYLOAD = {
    "is_tv_subscriber": True,
    "is_movie_package_subscriber": False,
    "subscription_age": 2,
    "bill_avg": 20,
    "remaining_contract": 0.5,
    "service_failure_count": 0,
    "download_avg": 10.0,
    "upload_avg": 2.0,
    "download_over_limit": 0,
}


class FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, _X):
        return [self.prediction]


app.app.dependency_overrides[get_model] = lambda: FakeModel(1)
client= TestClient(app.app)


def test_health():
    response= client.get("/health")
    assert response.status_code ==200
    assert response.json()== {"status": "healthy"}

def test_single_prediction_returns_model_output():
    response = client.post("/single_prediction", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {"churn_prediction": 1}

def test_single_prediction_rejects_invalid_input():
    bad_payload = {**VALID_PAYLOAD, "bill_avg": -5}
    response = client.post("/single_prediction", json=bad_payload)
    assert response.status_code == 422
