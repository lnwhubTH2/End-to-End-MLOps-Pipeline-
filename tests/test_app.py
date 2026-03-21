from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Iris Model API"}

def test_predict():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    result = response.json()
    assert "prediction_class" in result
    assert "confidence" in result
    assert isinstance(result["prediction_class"], int)
    assert isinstance(result["confidence"], float)
