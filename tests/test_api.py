"""REST API tests (run with: pytest)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)
GOOD = dict(cement=280, slag=0, fly_ash=0, water=190, superplasticizer=0, coarse_agg=970, fine_agg=800, age=28)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200 and "test_metrics" in r.json()


def test_predict_ok():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 5 < body["strength_mpa"] < 90
    assert body["low_mpa"] <= body["strength_mpa"] <= body["high_mpa"]


def test_predict_out_of_range_is_422_with_message():
    r = client.post("/predict", json={**GOOD, "cement": 9999})
    assert r.status_code == 422
    assert "outside the range" in str(r.json())


def test_predict_missing_field_is_422():
    bad = dict(GOOD); del bad["age"]
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_wrong_type_is_422():
    assert client.post("/predict", json={**GOOD, "water": "plenty"}).status_code == 422
